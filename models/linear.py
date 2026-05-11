"""线性模型：等权、ICIR 加权、Ridge 回归"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .base import ModelBase, cross_section_normalize
from factor.ic_compute import calc_ic_series, calc_icir


class EqualWeight(ModelBase):
    """等权合成：各因子截面标准化后等权平均"""

    def __init__(self):
        super().__init__(name="EqualWeight")

    def fit(self, factor_df: pd.DataFrame, forward_return: pd.Series) -> None:
        self._fitted = True

    def predict(self, factor_df: pd.DataFrame) -> pd.Series:
        normed = cross_section_normalize(factor_df)
        signal = normed.mean(axis=1)
        return cross_section_normalize(signal.to_frame("pred"))["pred"]


class ICIRWeighted(ModelBase):
    """ICIR 加权合成：每个因子的权重 = |ICIR|（基于训练期计算）"""

    def __init__(self, trim_quantile: tuple = (0.01, 0.99), min_samples: int = 5):
        super().__init__(name="ICIRWeighted")
        self.trim_quantile = trim_quantile
        self.min_samples = min_samples
        self.weights: dict = {}

    def fit(self, factor_df: pd.DataFrame, forward_return: pd.Series) -> None:
        for col in factor_df.columns:
            ic_df = calc_ic_series(
                factor=factor_df[col],
                forward_ret=forward_return,
                by_date=True,
                trim_quantile=self.trim_quantile,
                min_samples=self.min_samples,
            )
            if ic_df.empty:
                self.weights[col] = 0.0
                continue
            metrics = calc_icir(ic_df)
            self.weights[col] = abs(metrics["ICIR"])
        self._fitted = True

    def predict(self, factor_df: pd.DataFrame) -> pd.Series:
        normed = cross_section_normalize(factor_df)
        available = [c for c in self.weights if c in normed.columns]
        if not available:
            raise ValueError("没有可用的因子列")
        weighted = sum(self.weights[c] * normed[c] for c in available)
        total_w = sum(abs(self.weights[c]) for c in available) + 1e-12
        signal = weighted / total_w
        return cross_section_normalize(signal.to_frame("pred"))["pred"]


class RidgeRegression(ModelBase):
    """截面 Ridge 回归：用因子值拟合下期收益，正则化防止过拟合"""

    def __init__(self, alpha: float = 1.0):
        super().__init__(name="RidgeRegression")
        self.alpha = alpha
        self.model: Ridge = None
        self.feature_names: list = []

    def fit(self, factor_df: pd.DataFrame, forward_return: pd.Series) -> None:
        X = cross_section_normalize(factor_df)
        aligned = pd.concat([X, forward_return.rename("_ret_")], axis=1).dropna()
        self.feature_names = list(X.columns)
        if len(aligned) < len(self.feature_names) + 5:
            # 样本太少，退化为等权
            self.model = None
        else:
            self.model = Ridge(alpha=self.alpha, fit_intercept=False)
            self.model.fit(aligned[self.feature_names], aligned["_ret_"])
        self._fitted = True

    def predict(self, factor_df: pd.DataFrame) -> pd.Series:
        X = cross_section_normalize(factor_df)
        common = [c for c in self.feature_names if c in X.columns]
        if self.model is None:
            signal = X[common].mean(axis=1)
        else:
            coef = pd.Series(self.model.coef_, index=self.feature_names)
            signal = sum(coef[c] * X[c] for c in common)
        return cross_section_normalize(signal.to_frame("pred"))["pred"]
