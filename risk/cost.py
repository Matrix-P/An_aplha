"""交易成本模型：佣金、印花税、滑点"""
import pandas as pd
import numpy as np
from .base import RiskComponent


class Commission(RiskComponent):
    """佣金 + 印花税

    A股市场：佣金 ~0.025%（万2.5）, 印花税 0.05%（卖出收取）

    在 context 中需要提供 'prev_weights'（上期权重）来计算换手。
    不直接修改权重，而是返回调整后的权重（扣除买入成本后）。
    实际成本从 portfolio 层面扣除更合理，这里用于限制过度交易。
    """

    def __init__(self, commission_rate: float = 0.00025, stamp_tax: float = 0.0005,
                 max_turnover_cost_pct: float = 0.01):
        super().__init__(name="Commission")
        self.commission_rate = commission_rate
        self.stamp_tax = stamp_tax
        self.max_turnover_cost_pct = max_turnover_cost_pct

    def apply(self, weights: pd.Series, context: dict = None) -> pd.Series:
        if context is None or "prev_weights" not in context:
            return weights

        prev = context["prev_weights"]
        w = weights.copy()
        turnover = (w - prev).abs().sum()
        cost = turnover * self.commission_rate * 2  # 双边佣金
        sell_turnover = (prev - w).clip(lower=0).sum()
        cost += sell_turnover * self.stamp_tax  # 卖出印花税

        if cost > self.max_turnover_cost_pct:
            # 成本太高，缩减换手
            scale = self.max_turnover_cost_pct / (cost + 1e-12)
            w = prev + (w - prev) * scale

        return w

    def estimate_cost(self, weights: pd.Series, prev_weights: pd.Series) -> float:
        """估算本次调仓成本（占资金比例）"""
        turnover = (weights - prev_weights).abs().sum()
        cost = turnover * self.commission_rate * 2
        sell = (prev_weights - weights).clip(lower=0).sum()
        cost += sell * self.stamp_tax
        return cost


class Slippage(RiskComponent):
    """滑点模型：权重越大，冲击成本越高

    简单线性模型：slippage = base_bps + k * |weight_change| / sqrt(volume)

    context 需提供 'volumes' 或直接降权。
    """

    def __init__(self, max_weight_adjustment: float = 0.02):
        super().__init__(name="Slippage")
        self.max_weight_adjustment = max_weight_adjustment

    def apply(self, weights: pd.Series, context: dict = None) -> pd.Series:
        if context is None or "prev_weights" not in context:
            return weights

        prev = context["prev_weights"]
        w = weights.copy()
        change = (w - prev).abs()
        # 对换手过大的股票降权
        for s in change.index:
            if change[s] > self.max_weight_adjustment:
                sign = 1 if w[s] > prev.get(s, 0) else -1
                w[s] = prev.get(s, 0) + sign * self.max_weight_adjustment
        return w
