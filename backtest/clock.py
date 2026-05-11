"""统一时钟：按最细频率生成时间轴，标注每个 tick 的类型"""
from typing import List, Dict, Any
import pandas as pd
from .config import BacktestConfig

# freq 到 pandas 兼容格式的映射
FREQ_MAP = {
    "daily": "B",
    "60min": "60min",
    "30min": "30min",
    "1min": "1min",
}


def generate_timeline(config: BacktestConfig) -> pd.DataFrame:
    """生成统一时间轴，每行一个 tick

    Returns
    -------
    pd.DataFrame
        columns: time, is_decision, is_data, is_risk_check, date_label
    """
    fine_freq = FREQ_MAP.get(config.data_freq, "B")

    # 生成最细粒度的时间轴
    if config.data_freq == "daily":
        ticks = pd.date_range(config.start_date, config.end_date, freq="B")
    else:
        start = pd.to_datetime(config.start_date + " 09:30:00")
        end = pd.to_datetime(config.end_date + " 15:00:00")
        ticks = pd.date_range(start, end, freq=fine_freq)
        # 只保留 A 股交易时段
        ticks = ticks[
            ((ticks.hour == 9) & (ticks.minute >= 30)) |
            (ticks.hour == 10) |
            ((ticks.hour == 11) & (ticks.minute <= 30)) |
            (ticks.hour >= 13) & (ticks.hour < 15)
        ]

    df = pd.DataFrame({"time": ticks})
    df["date_label"] = df["time"].dt.strftime("%Y-%m-%d")
    df["time_label"] = df["time"].dt.strftime("%Y-%m-%d %H:%M")

    # 标注决策点
    df["is_decision"] = _mark_ticks(df["time"], config.decision_freq)
    # 标注风控巡检点
    df["is_risk_check"] = _mark_ticks(df["time"], config.risk_check_freq)
    # 每个 tick 都是数据点
    df["is_data"] = True

    return df


def _mark_ticks(times: pd.Series, freq: str) -> pd.Series:
    """标注哪些 tick 属于某频率的触发点"""
    if freq == "daily":
        result = pd.Series(False, index=times.index)
        seen_dates = set()
        for i, t in times.items():
            d = t.strftime("%Y-%m-%d")
            if d not in seen_dates:
                result.iloc[i] = True
                seen_dates.add(d)
        return result
    # 日内频率：每个 tick 都是触发点（时间轴已按该频率生成）
    return pd.Series(True, index=times.index)
