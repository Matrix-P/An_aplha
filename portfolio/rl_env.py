"""RL 交易环境：嵌入风控组件，提供 gym-like 接口"""
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List, Optional, Tuple
from models.base import ModelBase
from risk.base import RiskComponent


class RLEnvironment:
    """用于训练 RL Portfolio 的交易环境

    状态: 模型预测值 + 因子值 → (n_stocks, feature_dim) 矩阵
    动作: softmax 后的股票权重
    奖励: 组合收益 - 回撤惩罚
    """

    def __init__(self, models: List[ModelBase],
                 risk_components: List[RiskComponent],
                 dates, factor_df: pd.DataFrame,
                 ret_series: pd.Series,
                 feature_cols: Optional[List[str]] = None,
                 initial_capital: float = 1.0):
        self.models = models
        self.risk_components = risk_components
        self.dates = list(dates)
        self.factor_df = factor_df
        self.ret_series = ret_series
        self.feature_cols = feature_cols or list(factor_df.columns)
        self.initial_capital = initial_capital

        self.current_idx = 0
        self.positions = None
        self.portfolio_value = initial_capital
        self.peak_value = initial_capital

    @property
    def feature_dim(self) -> int:
        return len(self.models) + len(self.feature_cols)

    def reset(self) -> np.ndarray:
        self.current_idx = 0
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.positions = None
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        date = self.dates[self.current_idx]
        try:
            cross = self.factor_df.xs(date, level=0)
        except KeyError:
            return np.zeros((1, self.feature_dim))

        stocks = cross.index.tolist()
        features = []

        for m in self.models:
            try:
                pred = m.predict(cross)
                features.append(pred.values)
            except Exception:
                features.append(np.zeros(len(stocks)))

        for col in self.feature_cols:
            if col in cross.columns:
                features.append(cross[col].values)
            else:
                features.append(np.zeros(len(stocks)))

        state = np.column_stack(features).astype(np.float32)
        return np.nan_to_num(state, 0.0)

    def step(self, weights: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        date = self.dates[self.current_idx]
        try:
            cross = self.factor_df.xs(date, level=0)
        except KeyError:
            return self._get_state(), 0.0, True, {}

        stocks = cross.index.tolist()
        w = pd.Series(weights, index=stocks)

        # 风控链
        context = {
            "prev_weights": self.positions if self.positions is not None
            else pd.Series(0.0, index=stocks),
            "portfolio_value": self.portfolio_value,
            "initial_capital": self.initial_capital,
        }
        for comp in self.risk_components:
            w = comp.apply(w, context)

        # 只做多
        pos = w[w > 0]
        if pos.sum() > 0:
            w[pos.index] = pos / pos.sum()
        w[w < 0] = 0

        self.current_idx += 1
        if self.current_idx >= len(self.dates) - 1:
            done = True
            reward = 0.0
        else:
            done = False
            next_date = self.dates[self.current_idx]
            try:
                next_ret = self.ret_series.xs(next_date, level=0)
            except KeyError:
                reward = 0.0
            else:
                matched = [s for s in w.index if s in next_ret.index]
                reward = (w[matched].values * next_ret[matched].values).sum()
                self.portfolio_value *= (1 + reward)
                if self.portfolio_value < self.peak_value:
                    dd = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-12)
                    reward -= 0.01 * dd
                else:
                    self.peak_value = self.portfolio_value

        self.positions = w
        next_state = self._get_state() if not done else np.zeros_like(self._get_state())

        return next_state, reward, done, {"date": date, "portfolio_value": self.portfolio_value}
