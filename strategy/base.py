"""策略基类：纯信号生成，不关心仓位和风控"""
from abc import ABC, abstractmethod
from typing import List
import pandas as pd
from models.base import ModelBase


class StrategyBase(ABC):
    """纯信号策略：输入模型列表，输出全市场股票评分

    策略只管"哪只股票好"，不管"买多少、能不能买"。
    仓位管理和风控交给 Portfolio 层。
    """

    def __init__(self, models: List[ModelBase], name: str = "strategy"):
        self.models = models
        self.name = name

    @abstractmethod
    def score_stocks(self, date, factor_df: pd.DataFrame) -> pd.Series:
        """给定日期和因子，返回所有股票的综合评分（z-score）

        Returns
        -------
        pd.Series, index=stock_code, values=z-score（越高越好）
        """
        ...

    def _combine_predictions(self, factor_df: pd.DataFrame) -> pd.Series:
        """多模型预测等权平均 → z-score"""
        if not self.models:
            return pd.Series(dtype=float)
        preds = [m.predict(factor_df) for m in self.models]
        combined = sum(preds) / len(preds)
        # 截面标准化
        mean = combined.mean()
        std = combined.std()
        if std > 1e-12:
            return (combined - mean) / std
        return combined - mean
