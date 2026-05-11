"""暴露控制：行业中性、净敞口限制"""
import pandas as pd
import numpy as np
from .base import RiskComponent


class NetExposureLimit(RiskComponent):
    """净敞口限制：多头 vs 空头的不平衡度上限

    net_exposure = 多权重 - 空权重，限制在 [-limit, limit]
    """

    def __init__(self, max_net_exposure: float = 0.3):
        super().__init__(name="NetExposureLimit")
        self.max_net_exposure = max_net_exposure

    def apply(self, weights: pd.Series, context: dict = None) -> pd.Series:
        w = weights.copy()
        net = w.sum()
        if net > self.max_net_exposure:
            long_side = w[w > 0]
            short_side = w[w < 0]
            if len(long_side) > 0:
                scale = self.max_net_exposure / (net + 1e-12)
                w[long_side.index] *= scale
        elif net < -self.max_net_exposure:
            long_side = w[w > 0]
            short_side = w[w < 0]
            if len(short_side) > 0:
                scale = self.max_net_exposure / (abs(net) + 1e-12)
                w[short_side.index] *= scale
        return w


class SectorNeutral(RiskComponent):
    """行业中性：每个行业内权重和接近 0

    context 需提供 'sectors': pd.Series, index=stock_code, value=sector
    """

    def __init__(self):
        super().__init__(name="SectorNeutral")

    def apply(self, weights: pd.Series, context: dict = None) -> pd.Series:
        if context is None or "sectors" not in context:
            return weights

        sectors = context["sectors"]
        w = weights.copy()
        for sector in sectors.unique():
            mask = sectors[sectors == sector].index
            mask = [s for s in mask if s in w.index]
            if len(mask) == 0:
                continue
            sector_w = w[mask]
            w[mask] = sector_w - sector_w.mean()
        return w


class SizeNeutral(RiskComponent):
    """规模中性：大/小市值权重和接近 0

    context 需提供 'market_cap': pd.Series, index=stock_code
    """

    def __init__(self, n_bins: int = 2):
        super().__init__(name="SizeNeutral")
        self.n_bins = n_bins

    def apply(self, weights: pd.Series, context: dict = None) -> pd.Series:
        if context is None or "market_cap" not in context:
            return weights

        mcap = context["market_cap"]
        common = [s for s in mcap.index if s in weights.index]
        if len(common) < self.n_bins * 2:
            return weights

        bins = pd.qcut(mcap[common], self.n_bins, labels=False)
        w = weights.copy()
        for b in bins.unique():
            members = bins[bins == b].index
            members = [s for s in members if s in w.index]
            bin_w = w[members]
            w[members] = bin_w - bin_w.mean()
        return w
