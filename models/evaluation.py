"""模型评估：IC、分组收益、换手率"""
import numpy as np
import pandas as pd

from factor.ic_compute import calc_ic_series, calc_icir


def evaluate_model(
    prediction: pd.Series,
    forward_return: pd.Series,
    trim_quantile: tuple = (0.01, 0.99),
    min_samples: int = 5,
    n_groups: int = 5,
) -> dict:
    """
    全面评估模型预测质量。

    Parameters
    ----------
    prediction : pd.Series
        模型预测值，MultiIndex (date, stock)
    forward_return : pd.Series
        下期实际收益，MultiIndex (date, stock)
    trim_quantile : tuple
        收益截尾分位数
    min_samples : int
        每日最少样本数
    n_groups : int
        分组收益组数

    Returns
    -------
    dict
        含 IC、ICIR、分组收益等指标
    """
    ic_df = calc_ic_series(
        factor=prediction,
        forward_ret=forward_return,
        by_date=True,
        trim_quantile=trim_quantile,
        min_samples=min_samples,
    )
    if ic_df.empty:
        return {"error": "IC 序列为空"}

    metrics = calc_icir(ic_df)

    # 分组收益
    group_returns = _calc_group_returns(prediction, forward_return, n_groups, min_samples)

    # 换手率（双边）
    turnover = _calc_turnover(prediction)

    return {
        "mean_IC": metrics["mean_IC"],
        "std_IC": metrics["std_IC"],
        "ICIR": metrics["ICIR"],
        "mean_RankIC": metrics["mean_RankIC"],
        "RankICIR": metrics["RankICIR"],
        "ic_df": ic_df,
        "group_returns": group_returns,
        "turnover": turnover,
    }


def _calc_group_returns(
    prediction: pd.Series,
    forward_return: pd.Series,
    n_groups: int,
    min_samples: int,
) -> pd.DataFrame:
    """按预测值分组，计算每组平均收益"""
    aligned = pd.concat(
        [prediction.rename("pred"), forward_return.rename("ret")], axis=1
    ).dropna()

    # 按日期分组，每个截面内再按预测值分组
    results = []
    for date, group in aligned.groupby(level=0):
        if len(group) < n_groups:
            continue
        group["bucket"] = pd.qcut(
            group["pred"].rank(method="first"), n_groups, labels=False, duplicates="drop"
        )
        bucket_ret = group.groupby("bucket")["ret"].mean()
        bucket_ret.name = date
        results.append(bucket_ret)

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


def _calc_turnover(prediction: pd.Series, top_pct: float = 0.2) -> float:
    """计算双边换手率（top_pct 分组的日度平均换手）"""
    if not isinstance(prediction.index, pd.MultiIndex):
        return 0.0

    dates = sorted(prediction.index.get_level_values(0).unique())
    turnovers = []
    prev_top = None
    for date in dates:
        cross = prediction.loc[date]
        threshold = cross.quantile(1 - top_pct)
        current_top = set(cross[cross >= threshold].index)
        if prev_top is not None:
            intersection = len(current_top & prev_top)
            union = len(current_top | prev_top) or 1
            turnover = 1 - intersection / union
            turnovers.append(turnover)
        prev_top = current_top

    return np.mean(turnovers) if turnovers else 0.0


def compare_models(
    predictions: dict[str, pd.Series],
    forward_return: pd.Series,
    **kwargs,
) -> pd.DataFrame:
    """比较多个模型的评估指标，返回汇总 DataFrame"""
    rows = []
    for name, pred in predictions.items():
        result = evaluate_model(pred, forward_return, **kwargs)
        rows.append({
            "模型": name,
            "ICIR": result.get("ICIR", np.nan),
            "RankICIR": result.get("RankICIR", np.nan),
            "mean_IC": result.get("mean_IC", np.nan),
            "换手率": result.get("turnover", np.nan),
        })
    return pd.DataFrame(rows).set_index("模型")
