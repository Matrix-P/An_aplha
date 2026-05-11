"""订单执行模拟：手续费 + 成交量约束的流动性"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class Broker:
    """模拟订单执行，包含手续费和流动性约束

    流动性模型：对于每只股票，该 tick 最大可成交比例 ~ N(mean_frac, std_frac)
    截断到 [0, 1]。如果订单超出可成交量，只部分成交。
    """

    def __init__(self, commission_rate: float = 0.00025,
                 stamp_tax: float = 0.0005,
                 liquidity_enabled: bool = True,
                 liquidity_mean_frac: float = 0.5,
                 liquidity_std_frac: float = 0.25,
                 seed: Optional[int] = 42):
        self.commission_rate = commission_rate
        self.stamp_tax = stamp_tax
        self.liquidity_enabled = liquidity_enabled
        self.mean_frac = liquidity_mean_frac
        self.std_frac = liquidity_std_frac
        self.rng = np.random.RandomState(seed)

    def execute(self, target_weights: pd.Series,
                current_positions: pd.Series,
                prices: pd.Series,
                volumes: Optional[pd.Series] = None,
                portfolio_value: float = 1.0) -> Dict:
        """执行订单，返回成交结果

        Parameters
        ----------
        target_weights : pd.Series
            目标权重, index=stock_code
        current_positions : pd.Series
            当前持仓权重, index=stock_code
        prices : pd.Series
            当前价格, index=stock_code
        volumes : pd.Series, optional
            当前 tick 成交量（股）, index=stock_code。None 时不启用流动性约束
        portfolio_value : float
            当前组合总价值

        Returns
        -------
        dict
            filled_weights: 实际成交权重
            filled_values: 实际成交金额
            costs: 总交易成本
            fill_rates: 每只股票的成交率
            unfilled: 未成交的权重差额
        """
        target = target_weights.reindex(prices.index).fillna(0.0)
        current = current_positions.reindex(prices.index).fillna(0.0)
        change = target - current

        # 计算流动性约束下的最大可调权重
        max_change = pd.Series(np.inf, index=prices.index)
        if self.liquidity_enabled and volumes is not None and len(volumes) > 0:
            max_change = self._liquidity_limit(volumes, prices, portfolio_value)

        # 受限的实际权重变化
        actual_change = change.copy()
        for s in change.index:
            if abs(change[s]) > max_change[s]:
                actual_change[s] = np.sign(change[s]) * max_change[s]

        actual_weights = current + actual_change
        actual_values = actual_weights * portfolio_value

        # 交易成本
        turnover = actual_change.abs().sum() * portfolio_value
        sell_amount = (actual_change[actual_change < 0].abs().sum()) * portfolio_value
        costs = turnover * self.commission_rate + sell_amount * self.stamp_tax

        # 成交率
        fill_rates = pd.Series(1.0, index=change.index)
        for s in change.index:
            if abs(change[s]) > 1e-12:
                fill_rates[s] = abs(actual_change[s]) / abs(change[s])

        return {
            "filled_weights": actual_weights,
            "filled_values": actual_values,
            "costs": costs,
            "fill_rates": fill_rates,
            "unfilled": change - actual_change,
        }

    def _liquidity_limit(self, volumes: pd.Series, prices: pd.Series,
                         portfolio_value: float) -> pd.Series:
        """计算每只股票该 tick 的最大可调权重"""
        limits = pd.Series(np.inf, index=prices.index)
        for s in prices.index:
            if s not in volumes.index or volumes[s] <= 0:
                continue
            v = volumes[s]
            # max fillable shares ~ N(v * mean_frac, v * std_frac), 截断 [0, v]
            max_shares = self.rng.normal(v * self.mean_frac, v * self.std_frac)
            max_shares = max(0.0, min(max_shares, float(v)))
            max_value = max_shares * prices[s]
            limits[s] = max_value / (portfolio_value + 1e-12)
        return limits
