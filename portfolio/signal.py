"""信号驱动 Portfolio：依赖策略评分，转换为权重"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from portfolio.base import PortfolioBase
from strategy.base import StrategyBase
from risk.base import RiskComponent


class SignalPortfolio(PortfolioBase):
    """依赖策略评分的仓位管理器

    支持三种分配模式：
    - 'top_n': 买评分最高的 top_n 只
    - 'weighted': 按评分绝对值加权（正多负空）
    - 'quantile': top_q 做多 + bottom_q 做空

    必须传入一个 Strategy。
    """

    def __init__(self, strategy: StrategyBase,
                 risk_components: list,
                 mode: str = "top_n",
                 top_n: int = 20,
                 top_quantile: float = 0.2,
                 bottom_quantile: float = 0.2,
                 entry_threshold: float = 0.5,
                 exit_threshold: float = -1.0,
                 name="SignalPortfolio"):
        super().__init__(risk_components, entry_threshold, exit_threshold, name)
        self.strategy = strategy
        self.mode = mode
        self.top_n = top_n
        self.top_quantile = top_quantile
        self.bottom_quantile = bottom_quantile

    def step(self, date, factor_df: pd.DataFrame,
             context: Dict[str, Any]) -> pd.Series:
        # 1. 策略给全市场打分
        scores = self.strategy.score_stocks(date, factor_df)
        if len(scores) == 0:
            return pd.Series(dtype=float)

        # 2. 更新持仓池
        self._update_pool(scores)

        # 3. 信号 → 权重（只在池内分配）
        weights = self._scores_to_weights(scores)

        # 4. 过风控链
        context["prev_weights"] = pd.Series(self.positions,
                                             index=weights.index).fillna(0)
        weights = self._apply_risk_chain(weights, context)

        # 5. 清理零仓位
        self._cleanup_pool(weights)

        return weights

    def _scores_to_weights(self, scores: pd.Series) -> pd.Series:
        """将评分转为原始权重"""
        pool_scores = scores[scores.index.isin(self.pool)]

        if self.mode == "top_n":
            n = min(self.top_n, len(pool_scores))
            top = pool_scores.nlargest(n)
            w = pd.Series(0.0, index=scores.index)
            w[top.index] = 1.0 / n
            return w

        elif self.mode == "weighted":
            w = pd.Series(0.0, index=scores.index)
            pos = pool_scores[pool_scores > 0]
            neg = pool_scores[pool_scores < 0]
            if len(pos) > 0:
                w[pos.index] = pos / pos.sum()
            if len(neg) > 0:
                w[neg.index] = neg / abs(neg.sum())
            return w

        elif self.mode == "quantile":
            n = len(pool_scores)
            if n < 5:
                return pd.Series(0.0, index=scores.index)
            top_n = max(1, int(n * self.top_quantile))
            bottom_n = max(1, int(n * self.bottom_quantile))
            long_stocks = pool_scores.nlargest(top_n).index
            short_stocks = pool_scores.nsmallest(bottom_n).index
            w = pd.Series(0.0, index=scores.index)
            w[long_stocks] = 1.0 / top_n
            w[short_stocks] = -1.0 / bottom_n
            return w

        else:
            raise ValueError(f"Unknown mode: {self.mode}")
