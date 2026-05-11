"""模型基类"""
from abc import ABC, abstractmethod
from typing import Optional
import pickle
import pandas as pd


class ModelBase(ABC):
    """所有模型的统一接口

    输入: factor_df  — MultiIndex (date, stock) DataFrame, columns = factor names
    输出: prediction — MultiIndex (date, stock) Series, z-score
    """

    def __init__(self, name: str = "model"):
        self.name = name
        self._fitted = False

    @abstractmethod
    def fit(self, factor_df: pd.DataFrame, forward_return: pd.Series) -> None:
        ...

    @abstractmethod
    def predict(self, factor_df: pd.DataFrame) -> pd.Series:
        ...

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ModelBase":
        with open(path, "rb") as f:
            return pickle.load(f)


def cross_section_normalize(factor_df: pd.DataFrame) -> pd.DataFrame:
    """按日期横截面 z-score 标准化（去均值除标准差）"""
    if isinstance(factor_df.index, pd.MultiIndex):
        return factor_df.groupby(level=0).transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-12)
        )
    return (factor_df - factor_df.mean()) / (factor_df.std() + 1e-12)
