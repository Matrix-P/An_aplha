"""分位数策略：多模型评分等权平均"""
import pandas as pd
from strategy.base import StrategyBase


class QuantileStrategy(StrategyBase):
    """多模型评分等权平均，返回全市场 z-score"""

    def __init__(self, models, name="Quantile"):
        super().__init__(models, name)

    def score_stocks(self, date, factor_df: pd.DataFrame) -> pd.Series:
        if isinstance(factor_df.index, pd.MultiIndex):
            cross = factor_df.xs(date, level=0)
        else:
            cross = factor_df
        if len(cross) == 0:
            return pd.Series(dtype=float, name=date)
        return self._combine_predictions(cross)
