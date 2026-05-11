"""仓位约束：单票上限、总杠杆、最小持仓数"""
import pandas as pd
from .base import RiskComponent


class MaxPosition(RiskComponent):
    """单票仓位上限：任何一只股票的仓位不得超过 max_weight

    对多头和空头分别约束，超额按比例再分配到同侧其它股票。
    """

    def __init__(self, max_weight: float = 0.10):
        super().__init__(name="MaxPosition")
        self.max_weight = max_weight

    def _cap_side(self, w: pd.Series) -> pd.Series:
        """对单侧（全正或全负，取绝对值处理）进行上限约束"""
        if len(w) == 0:
            return w
        abs_w = w.abs()
        total = abs_w.sum()
        if total == 0:
            return w
        abs_w = abs_w / total
        for _ in range(20):
            over = abs_w[abs_w > self.max_weight]
            if len(over) == 0:
                break
            excess = (over - self.max_weight).sum()
            abs_w[over.index] = self.max_weight
            under = abs_w[abs_w < self.max_weight]
            if len(under) > 0:
                abs_w[under.index] += under / under.sum() * excess
        return w * (abs_w / w.abs())

    def apply(self, weights: pd.Series, context: dict = None) -> pd.Series:
        w = weights.copy()
        pos_mask = w > 0
        neg_mask = w < 0
        if pos_mask.any():
            w[pos_mask] = self._cap_side(w[pos_mask])
        if neg_mask.any():
            w[neg_mask] = -self._cap_side(-w[neg_mask])
        return w


class MaxLeverage(RiskComponent):
    """总杠杆上限：总权重的绝对值之和不得超过 max_leverage

    max_leverage=1.0 表示纯多头，不允许做空超配。
    """

    def __init__(self, max_leverage: float = 1.0):
        super().__init__(name="MaxLeverage")
        self.max_leverage = max_leverage

    def apply(self, weights: pd.Series, context: dict = None) -> pd.Series:
        w = weights.copy()
        total = w.abs().sum()
        if total > self.max_leverage:
            w = w * (self.max_leverage / total)
        return w


class MinPosition(RiskComponent):
    """最少持仓数：若持仓股票数不足，拒绝本次信号（返回空权重）"""

    def __init__(self, min_stocks: int = 5):
        super().__init__(name="MinPosition")
        self.min_stocks = min_stocks

    def apply(self, weights: pd.Series, context: dict = None) -> pd.Series:
        nonzero = (weights.abs() > 1e-8).sum()
        if nonzero < self.min_stocks:
            return pd.Series(0.0, index=weights.index)
        return weights


class EqualizeWeights(RiskComponent):
    """等权化：将非零权重标准化为等权（消除集中度风险）"""

    def __init__(self):
        super().__init__(name="EqualizeWeights")

    def apply(self, weights: pd.Series, context: dict = None) -> pd.Series:
        nonzero = weights[weights.abs() > 1e-8]
        if len(nonzero) == 0:
            return weights
        sign = nonzero.map(lambda x: 1 if x > 0 else -1)
        w = pd.Series(0.0, index=weights.index)
        w[nonzero.index] = sign / len(nonzero)
        return w
