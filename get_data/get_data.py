import os
import functools
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import numpy as np
import tushare as ts
from dotenv import load_dotenv


# ──── 通用区间合并/减法（支持任意 timedelta）────

def _merge_intervals(intervals: List[Tuple[str, str]],
                     gap: pd.Timedelta) -> List[Tuple[str, str]]:
    """合并重叠或相邻（间距 <= gap）的区间"""
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: pd.to_datetime(x[0]))
    merged = [list(sorted_iv[0])]
    for start, end in sorted_iv[1:]:
        prev_start, prev_end = merged[-1]
        if pd.to_datetime(start) <= pd.to_datetime(prev_end) + gap:
            merged[-1][1] = max(prev_end, end, key=lambda d: pd.to_datetime(d))
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _subtract_interval(full_start: str, full_end: str,
                       excluded: List[Tuple[str, str]],
                       gap: pd.Timedelta) -> List[Tuple[str, str]]:
    """从 [full_start, full_end] 中减去已覆盖区间，返回缺失区间

    对于日内数据（gap < 1天），返回 '%Y-%m-%d 00:00:00' 格式的完整日期时间；
    日线数据返回 '%Y-%m-%d' 格式。
    """
    excluded = sorted(excluded, key=lambda x: x[0])
    intraday = gap < pd.Timedelta("1D")
    fmt = '%Y-%m-%d %H:%M:%S' if intraday else '%Y-%m-%d'

    def _parse(s):
        dt = pd.to_datetime(s)
        return dt.floor('s')

    result = []
    cur = _parse(full_start)
    full_end_dt = _parse(full_end)
    for ex_s, ex_e in excluded:
        ex_s_dt = _parse(ex_s)
        ex_e_dt = _parse(ex_e)
        if ex_s_dt > cur:
            end_dt = ex_s_dt - gap
            if end_dt >= cur:
                result.append((cur.strftime(fmt), end_dt.strftime(fmt)))
        cur = max(cur, ex_e_dt + gap)
    if cur <= full_end_dt:
        result.append((cur.strftime(fmt), full_end_dt.strftime(fmt)))
    return result


# ──── 通用缓存装饰器 ────

def data_cache(cache_dir: str = "data/", freq: str = "daily",
               time_col: str = "trade_date", gap: str = "1D",
               file_format: str = "parquet"):
    """通用磁盘缓存装饰器，支持日线和分钟线

    Parameters
    ----------
    cache_dir : str
        缓存目录
    freq : str
        频率标签，用于文件命名: {ts_code}_{freq}_data.parquet
    time_col : str
        DataFrame 中的时间列名
    gap : str
        合并区间时允许的最大间隔，如 '1D'（日线）, '1h'（小时线）, '1min'（分钟线）
    file_format : str
        缓存文件格式 (parquet / pickle)
    """
    os.makedirs(cache_dir, exist_ok=True)
    gap_td = pd.Timedelta(gap)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(ts_code: str, start_date: str, end_date: str, **kwargs):
            # 日内数据：如果用户传的是纯日期，扩展为完整日区间
            intraday = gap_td < pd.Timedelta("1D")
            if intraday:
                if ' ' not in str(start_date):
                    start_date = f"{start_date} 00:00:00"
                if ' ' not in str(end_date):
                    end_date = f"{end_date} 23:59:59"

            data_file = os.path.join(cache_dir, f"{ts_code}_{freq}_data.{file_format}")
            meta_file = os.path.join(cache_dir, f"{ts_code}_{freq}_intervals.txt")

            # 加载已有数据
            existing_df = None
            if os.path.exists(data_file):
                try:
                    if file_format == "parquet":
                        existing_df = pd.read_parquet(data_file)
                    else:
                        existing_df = pd.read_pickle(data_file)
                    if time_col in existing_df.columns:
                        existing_df[time_col] = pd.to_datetime(existing_df[time_col])
                except Exception as e:
                    print(f"读取缓存失败: {e}")

            # 加载已覆盖区间
            intervals = []
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split(',')
                            if len(parts) >= 2:
                                intervals.append((parts[0], parts[1]))

            # 计算缺失区间
            missing = _subtract_interval(start_date, end_date, intervals, gap_td)
            if not missing:
                if existing_df is not None:
                    s_dt, e_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
                    mask = (existing_df[time_col] >= s_dt) & (existing_df[time_col] <= e_dt)
                    return existing_df[mask].copy()
                return pd.DataFrame()

            # 逐段拉取
            new_parts = []
            new_intervals = []
            fmt = '%Y-%m-%d %H:%M:%S' if freq != "daily" else '%Y-%m-%d'
            for miss_start, miss_end in missing:
                part_df = func(ts_code=ts_code, start_date=miss_start,
                               end_date=miss_end, **kwargs)
                if part_df is not None and not part_df.empty:
                    if time_col in part_df.columns:
                        part_df[time_col] = pd.to_datetime(part_df[time_col])
                    new_parts.append(part_df)
                    actual_start = part_df[time_col].min().strftime(fmt)
                    actual_end = part_df[time_col].max().strftime(fmt)
                    new_intervals.append((actual_start, actual_end))
                else:
                    # 该区间无数据，标记为已覆盖防止重复拉取
                    new_intervals.append((miss_start, miss_end))

            if not new_parts and existing_df is None:
                return pd.DataFrame()

            # 合并
            if existing_df is not None:
                all_df = pd.concat([existing_df] + new_parts, ignore_index=True)
            else:
                all_df = pd.concat(new_parts, ignore_index=True)
            all_df = all_df.drop_duplicates(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

            # 保存区间元数据
            merged_intervals = _merge_intervals(intervals + new_intervals, gap_td)
            with open(meta_file, 'w') as f:
                for s, e in merged_intervals:
                    f.write(f"{s},{e}\n")

            # 保存数据
            try:
                if file_format == "parquet":
                    all_df.to_parquet(data_file, index=False)
                else:
                    all_df.to_pickle(data_file)
            except Exception as e:
                print(f"写入缓存失败: {e}")

            # 返回请求区间
            s_dt, e_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
            mask = (all_df[time_col] >= s_dt) & (all_df[time_col] <= e_dt)
            return all_df[mask].copy()
        return wrapper
    return decorator


# ──── 日线数据（保持向后兼容）────

@data_cache(cache_dir="data", freq="daily", time_col="trade_date", gap="1D")
def fetch_daily_from_tushare(pro, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Tushare 日线 OHLCV，自动缓存"""
    start = start_date.replace('-', '')[:8]
    end = end_date.replace('-', '')[:8]
    df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
    if df is None or df.empty:
        return pd.DataFrame()
    return df[['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']]


# ──── 分钟线数据 ────

@data_cache(cache_dir="data", freq="1min", time_col="trade_time", gap="1min")
def fetch_minute_from_tushare(pro, ts_code: str, start_date: str,
                               end_date: str) -> pd.DataFrame:
    """Tushare 1分钟线 OHLCV，自动缓存

    返回列: trade_time, open, high, low, close, vol, amount
    """
    start = start_date.replace('-', '')[:8]
    end = end_date.replace('-', '')[:8]
    try:
        df = pro.stk_mins(ts_code=ts_code, freq='1min',
                          start_date=start, end_date=end)
    except AttributeError:
        raise RuntimeError(
            "Tushare 分钟线接口不可用，请确认:\n"
            "1. Tushare 版本 >= 1.4.0\n"
            "2. 账号有分钟线权限（需积分 >= 2000）\n"
            "3. 接口名为 pro.stk_mins()"
        )
    if df is None or df.empty:
        return pd.DataFrame()
    # Tushare 返回的分钟线可能包含 ts_code, trade_time, open, high, low, close, vol, amount
    cols = [c for c in ['trade_time', 'open', 'high', 'low', 'close', 'vol', 'amount']
            if c in df.columns]
    if 'trade_time' not in df.columns and 'trade_date' in df.columns:
        df = df.rename(columns={'trade_date': 'trade_time'})
    return df[cols] if cols else df


# ──── 小时线数据 ────

@data_cache(cache_dir="data", freq="60min", time_col="trade_time", gap="1h")
def fetch_hourly_from_tushare(pro, ts_code: str, start_date: str,
                               end_date: str) -> pd.DataFrame:
    """Tushare 60分钟线 OHLCV（1分钟聚合），自动缓存

    先从分钟线获取，再 resample 到 60 分钟 OHLCV。
    返回列: trade_time, open, high, low, close, vol, amount
    """
    # 拉取 1 分钟线（复用分钟线缓存）
    df_1min = fetch_minute_from_tushare(
        pro=pro, ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df_1min is None or df_1min.empty:
        return pd.DataFrame()

    df = df_1min.copy()
    df['trade_time'] = pd.to_datetime(df['trade_time'])
    df = df.set_index('trade_time')

    # 60 分钟 OHLCV 聚合
    hourly = df.resample('60min', label='right', closed='right').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'vol': 'sum',
        'amount': 'sum',
    }).dropna()

    hourly = hourly.reset_index()
    return hourly


# ──── API 初始化 ────

def get_api():
    load_dotenv('config/tushare.env')
    share_token = os.getenv('TUSHARE_TOKEN')
    return ts.pro_api(token=share_token)
