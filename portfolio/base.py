"""Portfolio 基类：持仓池管理 + 风控执行"""
from abc import ABC, abstractmethod
from typing import List, Set, Dict, Any
import pandas as pd
from risk.base import RiskComponent


class PortfolioBase(ABC):
    """仓位管理器：管理持仓池、执行风控、产出目标权重

    职责：
    1. 维护持仓候选池（哪些股票可以交易）
    2. 根据信号（或直接）生成目标权重
    3. 过风控链
    4. T 日闭盘后清理零仓位

    Parameters
    ----------
    risk_components : list of RiskComponent
        按序执行的风控组件链
    entry_threshold : float
        新股票入池的 z-score 阈值
    exit_threshold : float
        池内股票踢出的 z-score 阈值
    """

    def __init__(self, risk_components: List[RiskComponent],
                 entry_threshold: float = 0.5,
                 exit_threshold: float = -1.0,
                 name: str = "portfolio"):
        self.risk_components = risk_components
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.name = name

        self.pool: Set[str] = set()
        self.positions: Dict[str, float] = {}
        self.position_history: list = []

    @abstractmethod
    def step(self, date, factor_df: pd.DataFrame,
             context: Dict[str, Any]) -> pd.Series:
        """T 日闭盘后执行：返回 T+1 目标权重

        Returns
        -------
        pd.Series, index=stock_code, values=weight
        """
        ...

    def _apply_risk_chain(self, weights: pd.Series,
                          context: Dict[str, Any]) -> pd.Series:
        """执行风控链"""
        w = weights.copy()
        for comp in self.risk_components:
            w = comp.apply(w, context)
        return w

    def _update_pool(self, scores: pd.Series):
        """根据评分更新持仓池"""
        for stock, score in scores.items():
            in_pool = stock in self.pool
            has_position = self.positions.get(stock, 0) != 0

            if has_position:
                continue  # 有仓位保持不变
            elif in_pool and score > self.exit_threshold:
                continue  # 池内信号尚可，保留
            elif not in_pool and score > self.entry_threshold:
                self.pool.add(stock)  # 新高分入池
            elif in_pool and score <= self.exit_threshold:
                self.pool.discard(stock)  # 信号太差踢出

    def _cleanup_pool(self, weights: pd.Series):
        """清零仓股票移出持仓池"""
        self.pool = {s for s in self.pool
                     if s in weights.index and abs(weights.get(s, 0)) > 1e-8}
        self.positions = {s: w for s, w in weights.items() if abs(w) > 1e-8}
