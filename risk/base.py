"""风控组件基类"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class RiskComponent(ABC):
    """所有风控组件的统一接口

    每个组件接收权重向量 + 上下文，返回修正后的权重向量。
    策略层按列表顺序执行风控链。
    """

    def __init__(self, name: str = "risk"):
        self.name = name

    @abstractmethod
    def apply(self, weights: pd.Series, context: Dict[str, Any]) -> pd.Series:
        """
        Parameters
        ----------
        weights : pd.Series
            目标权重，index = stock_code, values = weight
        context : dict
            上下文信息，可含 'prices', 'positions', 'date', 'capital' 等

        Returns
        -------
        pd.Series
            修正后权重
        """
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"
