# factor_compute.py

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Union, Optional
from datetime import datetime, timedelta

# 导入已有的模块
from get_data import fetch_daily_from_tushare   # 获取数据的函数
from .factor_compiler import compile_expression, _operators  # 表达式编译函数

def extract_max_lookback(expr: str) -> int:
    """
    从因子表达式字符串中提取所需的最大历史回溯天数。
    支持算子：MA, SMA, EMA, Mean, Std, Sum, Max, Min, Rank, Ref, Delta, Corr, Cov, RSI 等。
    返回最大窗口数值，若无窗口则返回0。
    """
    # 常见窗口函数模式：函数名(..., 数值) 或 函数名(数值)
    # 我们简单使用正则匹配数字，但要避免匹配到其他数字（如常数）。
    # 更精确的方法是解析AST，但为简化，我们匹配常见算子后的数字。
    # 为了提高可靠性，我们匹配形如 "FuncName(..., N)" 的模式。
    patterns = [
        r'(?:MA|SMA|EMA|Mean|Std|Sum|Max|Min|Rank|RSI|ATR)\s*\([^,]*,\s*(\d+)',   # 双参数
        r'(?:Ref|Delta)\s*\([^,]*,\s*(\d+)',                                        # Ref(x, d)
        r'Corr\s*\([^,]+,[^,]+,\s*(\d+)',                                          # Corr(x, y, n)
        r'Cov\s*\([^,]+,[^,]+,\s*(\d+)',                                           # Cov(x, y, n)
        r'Rolling\s*\([^,]+,\s*(\d+)',                                             # 通用滚动
    ]
    max_window = 0
    for pat in patterns:
        matches = re.findall(pat, expr, re.IGNORECASE)
        for m in matches:
            window = int(m)
            if window > max_window:
                max_window = window
    return max_window

def get_max_lookback_from_exprs(exprs: List[str]) -> int:
    """从多个表达式中获取最大回溯窗口"""
    max_window = 0
    for expr in exprs:
        w = extract_max_lookback(expr)
        if w > max_window:
            max_window = w
    return 2*max_window

def fetch_stock_data(
    pro,
    stock_pool: List[str],
    start_date: str,
    end_date: str,
    lookback_days: int = 0
) -> Dict[str, pd.DataFrame]:
    """
    获取股票池中所有股票的日线数据。
    - start_date: 用户需要的起始日期（格式 'YYYY-MM-DD'）
    - lookback_days: 额外往前取的天数（用于计算因子窗口）
    实际数据起始日期 = start_date - lookback_days
    """
    # 计算实际拉取起始日期（自然日减，后续交易日自动过滤）
    actual_start = pd.to_datetime(start_date) - timedelta(days=lookback_days)
    actual_start_str = actual_start.strftime('%Y-%m-%d')
    
    stock_data = {}
    for ts_code in stock_pool:
        # print(f"正在获取 {ts_code} 数据...")
        df = fetch_daily_from_tushare(
            pro = pro,
            ts_code=ts_code,
            start_date=actual_start_str,
            end_date=end_date
        )
        if df.empty:
            # print(f"警告: {ts_code} 无数据")
            continue
        # 确保日期列为 datetime 并排序
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').set_index('trade_date')
        stock_data[ts_code] = df
    return stock_data

def compute_factors_for_stock(
    stock_df: pd.DataFrame,
    factor_dict: Dict[str, str],
    start_date: str
) -> pd.DataFrame:
    """
    对一只股票的数据，计算所有因子表达式，返回从 start_date 开始的因子值 DataFrame。
    stock_df: 日线数据，index为日期，包含 open, high, low, close, volume 等列。
    factor_dict: 字典，键为因子名，值为表达式字符串，例如 {'ma5': 'SMA($close,5)', 'delta5': 'Delta($close,5)'}
    """
    # 为每个表达式编译函数
    compiled = {name: compile_expression(expr) for name, expr in factor_dict.items()}
    
    # 计算因子值
    factor_values = {}
    for name, expr in factor_dict.items():
        try:
            factor_series = compiled[name](stock_df)
            factor_values[name] = factor_series
        except Exception as e:
            print(f"计算表达式 {expr} 失败: {e}")
            factor_values[name] = pd.Series(index=stock_df.index, dtype=float)
    
    result = pd.DataFrame(factor_values, index=stock_df.index)
    # 截取 start_date 之后的数据（包含当天）
    result = result[result.index >= pd.to_datetime(start_date)]
    return result


def compute_factors_for_pool(
    pro,
    factor_dict: Dict[str, str],
    stock_pool: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    lookback_days: Optional[int] = None,
    auto_lookback: bool = True
) -> pd.DataFrame:
    """
    参数
    ----------
    factor_dict : Dict[str, str]
        因子字典，键为因子名，值为表达式字符串。
        例如 {'ma5': 'MA($close,5)', 'delta5': 'Delta($close,5)'}
    stock_pool : List[str]
        股票代码列表，格式 '600519.SH'
    start_date : str
        用户需要因子数据的起始日期（因子值从该日期开始有效）
    end_date : str, optional
        结束日期，默认为今天
    lookback_days : int, optional
        手动指定历史回溯天数（用于获取足够数据计算因子）。若不指定且 auto_lookback=True，则自动从表达式解析。
    auto_lookback : bool, default True
        是否自动解析最大回溯窗口。

    返回
    ----------
    pd.DataFrame
        长格式数据，列包括：trade_date, stock, 以及每个因子列（键名作为列名）
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # 提取表达式列表用于回溯窗口解析
    expr_list = list(factor_dict.values())
    
    # 确定回溯天数
    if lookback_days is None:
        if auto_lookback:
            lookback_days = get_max_lookback_from_exprs(expr_list)
            # print(f"自动解析最大回溯窗口为 {lookback_days} 天")
        else:
            lookback_days = 0
    
    # 获取所有股票数据
    stock_data = fetch_stock_data(pro, stock_pool, start_date, end_date, lookback_days)
    
    # 存储每只股票的因子结果
    all_factors = []
    for stock, df in stock_data.items():
        factor_df = compute_factors_for_stock(df, factor_dict, start_date)
        if factor_df.empty:
            continue
        # 转换为长格式：添加 stock 列
        factor_df = factor_df.reset_index().rename(columns={'index': 'trade_date'})
        factor_df['stock'] = stock
        all_factors.append(factor_df)
    
    if not all_factors:
        return pd.DataFrame()
    
    # 合并所有股票
    result = pd.concat(all_factors, ignore_index=True)
    # 确保日期为 datetime
    result['trade_date'] = pd.to_datetime(result['trade_date'])
    # 按日期和股票排序
    result = result.sort_values(['trade_date', 'stock']).reset_index(drop=True)
    result = result.set_index(['trade_date', 'stock'])
    return result