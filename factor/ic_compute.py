import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def align_and_truncate(
    factor: pd.Series,
    forward_ret: pd.Series,
    date_col: str = None,
    trim_quantile: tuple = None
):
    """
    对齐因子值与未来收益率，并截断无效数据。

    Parameters
    ----------
    factor : pd.Series
        因子值，索引应为日期（或包含日期列）。
    forward_ret : pd.Series
        未来收益率，索引应为日期（或包含日期列）。
    date_col : str, optional
        如果 Series 的索引不是日期，可以指定列名（需先合并为 DataFrame）。
    trim_quantile : tuple, optional
        例如 (0.01, 0.99)，剔除收益率低于1%分位和高于99%分位的极端值。

    Returns
    -------
    aligned_factor : pd.Series
        对齐后的因子值
    aligned_ret : pd.Series
        对齐后的收益率
    """
    # 转换为 DataFrame 以便对齐
    df = pd.DataFrame({'factor': factor, 'ret': forward_ret})
    if date_col is not None:
        # 如果提供了日期列，则按日期对齐（假设日期在列中）
        df = df.set_index(date_col)
    # 按索引对齐，dropna 会删除任意一列为 NaN 的行
    df = df.dropna()
    if trim_quantile is not None:
        lower, upper = trim_quantile
        q_low = df['ret'].quantile(lower)
        q_high = df['ret'].quantile(upper)
        df = df[(df['ret'] >= q_low) & (df['ret'] <= q_high)]
    return df['factor'], df['ret']


def calc_ic_series(
    factor: pd.Series,
    forward_ret: pd.Series,
    by_date: bool = True,
    date_col: str = None,
    trim_quantile: tuple = None,
    min_samples: int = 3
) -> pd.DataFrame:
    """
    计算每日/每期的 IC 和 RankIC，自动对齐并截断。

    Parameters
    ----------
    factor : pd.Series
        因子值，如果是横截面模式，索引应为 MultiIndex (trade_date, asset)；
        如果是时间序列模式，索引应为日期。
    forward_ret : pd.Series
        未来收益率，结构同上。
    by_date : bool
        True: 横截面模式（多股票），按日期分组计算。
        False: 时间序列模式（单股票），直接计算整个序列的相关系数。
    date_col : str, optional
        如果 Series 的索引不是日期，可以指定列名（仅当 by_date=True 时有效）。
    trim_quantile : tuple, optional
        剔除收益率极端分位数，如 (0.01, 0.99)。
    min_samples : int
        每个日期至少需要的样本数（横截面模式）。

    Returns
    -------
    pd.DataFrame
        包含 'trade_date', 'IC', 'RankIC' 的 DataFrame。
        若为时间序列模式，则返回单行。
    """
    if by_date:
        # 确保 factor 和 forward_ret 是同一 DataFrame 的列，以便分组
        df = pd.DataFrame({'factor': factor, 'ret': forward_ret})
        if date_col is None:
            # 尝试从索引中提取日期
            if isinstance(df.index, pd.MultiIndex) and 'trade_date' in df.index.names:
                df = df.reset_index()
                date_col = 'trade_date'
            elif isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                date_col = 'index'
            else:
                raise ValueError("横截面模式需要提供 date_col 或具有日期索引")
        # 按日期分组计算
        results = []
        for dt, group in df.groupby(date_col):
            # 对齐并截断
            f, r = align_and_truncate(group['factor'], group['ret'], trim_quantile=trim_quantile)
            if len(f) < min_samples:
                continue
            ic = f.corr(r)
            rank_ic = spearmanr(f, r)[0]
            results.append({'trade_date': dt, 'IC': ic, 'RankIC': rank_ic})
        return pd.DataFrame(results)
    else:
        # 时间序列模式：直接对齐整个序列
        f, r = align_and_truncate(factor, forward_ret, trim_quantile=trim_quantile)
        if len(f) < min_samples:
            return pd.DataFrame({'IC': [np.nan], 'RankIC': [np.nan]})
        ic = f.corr(r)
        rank_ic = spearmanr(f, r)[0]
        return pd.DataFrame({'IC': [ic], 'RankIC': [rank_ic]})


def calc_icir(
    ic_df: pd.DataFrame,
    periods_per_year: int = 252,
    ic_col: str = 'IC',
    rank_ic_col: str = 'RankIC'
) -> dict:
    """计算 ICIR 和 RankICIR（年化）"""
    ic_series = ic_df[ic_col].dropna()
    rank_series = ic_df[rank_ic_col].dropna()
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    icir = mean_ic / std_ic * (periods_per_year ** 0.5) if std_ic != 0 else np.nan
    mean_rank = rank_series.mean()
    std_rank = rank_series.std()
    rank_icir = mean_rank / std_rank * (periods_per_year ** 0.5) if std_rank != 0 else np.nan
    return {
        'mean_IC': mean_ic,
        'std_IC': std_ic,
        'ICIR': icir,
        'mean_RankIC': mean_rank,
        'std_RankIC': std_rank,
        'RankICIR': rank_icir,
        'count': len(ic_series)
    }