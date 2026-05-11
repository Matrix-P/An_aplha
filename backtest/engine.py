"""回测引擎：统一时钟 + 订单执行 + 风控巡检"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

from .config import BacktestConfig
from .clock import generate_timeline
from .broker import Broker


class BacktestEngine:
    """统一时钟回测引擎

    按最细粒度推进时间轴，在每个 tick：
    - 决策点: portfolio.step() 获取目标权重
    - 所有点: broker 执行订单 → 更新净值 → 记录

    Parameters
    ----------
    config : BacktestConfig
    portfolio : PortfolioBase
    """

    def __init__(self, config: BacktestConfig, portfolio):
        self.config = config
        self.portfolio = portfolio
        self.broker = Broker(
            commission_rate=config.commission_rate,
            stamp_tax=config.stamp_tax,
            liquidity_enabled=config.liquidity_enabled,
            liquidity_mean_frac=config.liquidity_mean_frac,
            liquidity_std_frac=config.liquidity_std_frac,
        )
        self.nav = config.initial_capital
        self.peak_nav = config.initial_capital
        self.timeline = None
        self.records: list = []

    def run(self, factor_df: pd.DataFrame,
            price_df: Optional[pd.DataFrame] = None,
            volume_df: Optional[pd.DataFrame] = None,
            ret_series: Optional[pd.Series] = None) -> pd.DataFrame:
        """执行回测

        Parameters
        ----------
        factor_df : pd.DataFrame
            MultiIndex (date, stock) 因子值
        price_df : pd.DataFrame, optional
            MultiIndex (date, stock) 价格，用于计算持仓价值
        volume_df : pd.DataFrame, optional
            MultiIndex (date_tick, stock) 成交量（股），用于流动性约束
        ret_series : pd.Series, optional
            MultiIndex (date, stock) 每 tick 收益率（因子层用的下期收益）

        Returns
        -------
        pd.DataFrame
            每日/每 tick 的详细记录
        """
        self.timeline = generate_timeline(self.config)
        self.nav = self.config.initial_capital
        self.peak_nav = self.config.initial_capital
        self.records = []

        positions = pd.Series(dtype=float)
        target_weights = pd.Series(dtype=float)
        current_date_label = None

        for idx, row in self.timeline.iterrows():
            tick_time = row["time"]
            date_label = row["date_label"]
            is_decision = row["is_decision"]
            is_risk_check = row["is_risk_check"]
            new_day = (date_label != current_date_label)
            if new_day:
                current_date_label = date_label

            # 1. 决策点：获取新的目标权重
            if is_decision:
                try:
                    context = {
                        "portfolio_value": self.nav,
                        "positions": positions.copy(),
                        "initial_capital": self.config.initial_capital,
                    }
                    target_weights = self.portfolio.step(
                        date=date_label, factor_df=factor_df, context=context)
                except Exception as e:
                    target_weights = pd.Series(dtype=float)

            # 2. 风控巡检：检查是否需要止损/熔断
            if is_risk_check and len(target_weights) > 0:
                context = {
                    "portfolio_value": self.nav,
                    "positions": positions,
                    "initial_capital": self.config.initial_capital,
                    "prev_weights": positions,
                }
                target_weights = self._run_risk_checks(target_weights, context)

            # 3. 执行订单
            if len(target_weights) > 0 and is_decision:
                prices = self._get_prices(tick_time, price_df, target_weights.index)
                volumes_tick = self._get_volumes(tick_time, volume_df, target_weights.index)

                result = self.broker.execute(
                    target_weights=target_weights,
                    current_positions=positions,
                    prices=prices,
                    volumes=volumes_tick,
                    portfolio_value=self.nav,
                )
                positions = result["filled_weights"]
                self.nav -= result["costs"]
            else:
                result = {
                    "filled_weights": positions,
                    "costs": 0.0,
                    "fill_rates": pd.Series(1.0, index=positions.index),
                    "unfilled": pd.Series(0.0, index=positions.index),
                }

            # 4. 计算收益（用 ret_series 或价格变动）
            period_return = 0.0
            if ret_series is not None and len(positions) > 0:
                try:
                    ret_cross = ret_series.xs(date_label, level=0)
                    matched = [s for s in positions.index if s in ret_cross.index]
                    if matched:
                        r = positions[matched].values * ret_cross[matched].values
                        r = r[~np.isnan(r)]
                        if len(r) > 0:
                            period_return = r.sum()
                except KeyError:
                    pass

            prev_nav = self.nav
            self.nav *= (1.0 + period_return)
            if self.nav > self.peak_nav:
                self.peak_nav = self.nav

            # 5. 记录
            self.records.append({
                "time": tick_time,
                "date": date_label,
                "is_decision": is_decision,
                "is_risk_check": is_risk_check,
                "nav": self.nav,
                "period_return": period_return,
                "costs": result["costs"],
                "positions": positions.to_dict(),
                "max_drawdown": (self.peak_nav - self.nav) / (self.peak_nav + 1e-12),
            })

        return pd.DataFrame(self.records)

    def _get_prices(self, tick_time, price_df, stocks) -> pd.Series:
        """获取某 tick 的价格"""
        if price_df is None:
            return pd.Series(1.0, index=stocks)
        date_str = pd.to_datetime(tick_time).strftime("%Y-%m-%d")
        try:
            if isinstance(price_df.index, pd.MultiIndex):
                cross = price_df.xs(date_str, level=0)
            else:
                cross = price_df
            return cross["close"] if "close" in cross.columns else cross.iloc[:, 0]
        except (KeyError, IndexError):
            return pd.Series(1.0, index=stocks)

    def _get_volumes(self, tick_time, volume_df, stocks) -> Optional[pd.Series]:
        """获取某 tick 的成交量"""
        if volume_df is None:
            return None
        date_str = pd.to_datetime(tick_time).strftime("%Y-%m-%d")
        try:
            if isinstance(volume_df.index, pd.MultiIndex):
                cross = volume_df.xs(date_str, level=0)
            else:
                cross = volume_df
            return cross["vol"] if "vol" in cross.columns else None
        except (KeyError, IndexError):
            return None

    def _run_risk_checks(self, weights, context):
        """运行需要日内巡检的风控组件（止损类）"""
        from risk.stoploss import TrailingStop, MaxDrawdownBlowout
        for comp in self.portfolio.risk_components:
            if isinstance(comp, (TrailingStop, MaxDrawdownBlowout)):
                weights = comp.apply(weights, context)
        return weights
