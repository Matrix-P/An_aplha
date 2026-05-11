"""止损/止盈：回撤止损、固定止盈、最大回撤熔断"""
import pandas as pd
import numpy as np
from .base import RiskComponent


class TrailingStop(RiskComponent):
    """回撤止损：自持仓以来回撤超过 max_drawdown 则清仓

    context 需要提供 'portfolio_value' 或 'cumulative_return'
    """

    def __init__(self, max_drawdown: float = 0.10):
        super().__init__(name="TrailingStop")
        self.max_drawdown = max_drawdown
        self._peak = None

    def apply(self, weights: pd.Series, context: dict = None) -> pd.Series:
        if context is None or "portfolio_value" not in context:
            return weights

        pv = context["portfolio_value"]
        if self._peak is None or pv > self._peak:
            self._peak = pv

        dd = (self._peak - pv) / (self._peak + 1e-12)
        if dd > self.max_drawdown:
            self._peak = None
            return pd.Series(0.0, index=weights.index)

        return weights


class MaxDrawdownBlowout(RiskComponent):
    """最大回撤熔断：自初始资金以来回撤超过 max_dd 则全部清仓

    context 需要提供 'portfolio_value' 和 'initial_capital'
    """

    def __init__(self, max_drawdown: float = 0.20, initial_capital: float = 1.0):
        super().__init__(name="MaxDrawdownBlowout")
        self.max_drawdown = max_drawdown
        self.initial_capital = initial_capital

    def apply(self, weights: pd.Series, context: dict = None) -> pd.Series:
        if context is None:
            return weights

        pv = context.get("portfolio_value", self.initial_capital)
        dd = (self.initial_capital - pv) / (self.initial_capital + 1e-12)
        if dd > self.max_drawdown:
            return pd.Series(0.0, index=weights.index)

        return weights


class TakeProfit(RiskComponent):
    """固定止盈：单只股票收益超过 target 则减半仓位"""

    def __init__(self, target_return: float = 0.20):
        super().__init__(name="TakeProfit")
        self.target_return = target_return

    def apply(self, weights: pd.Series, context: dict = None) -> pd.Series:
        if context is None or "stock_returns" not in context:
            return weights

        stock_rets = context["stock_returns"]
        w = weights.copy()
        for s in w.index:
            if s in stock_rets.index and stock_rets[s] > self.target_return:
                w[s] = w[s] * 0.5
        return w
