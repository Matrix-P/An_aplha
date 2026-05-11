"""绩效指标计算"""
import numpy as np
import pandas as pd


def compute_metrics(records: pd.DataFrame,
                    periods_per_year: int = 252) -> dict:
    """从回测记录计算全套绩效指标

    Parameters
    ----------
    records : pd.DataFrame
        backtest engine 输出的记录，必须含 nav 列
    periods_per_year : int
        年化周期数，日线 252，小时线约 1008

    Returns
    -------
    dict
    """
    if records.empty or "nav" not in records.columns:
        return {"error": "records 为空"}

    nav = records["nav"].values
    init_nav = nav[0]

    # 日收益序列（从决策点提取）
    if "is_decision" in records.columns:
        daily = records[records["is_decision"]].copy()
    else:
        daily = records.copy()

    returns = daily["nav"].pct_change().dropna().values
    if len(returns) < 2:
        return {"error": "数据点不足"}

    # 年化收益
    total_return = (nav[-1] / init_nav - 1)
    n_periods = len(returns)
    annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

    # 年化波动率
    annual_vol = returns.std() * np.sqrt(periods_per_year)

    # Sharpe
    sharpe = (annual_return - 0.02) / (annual_vol + 1e-12)  # 假设无风险 2%

    # 最大回撤
    peak = np.maximum.accumulate(nav)
    drawdowns = (peak - nav) / peak
    max_dd = drawdowns.max()

    # Calmar
    calmar = annual_return / (max_dd + 1e-12)

    # 胜率
    win_rate = (returns > 0).mean()

    # 盈亏比
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    profit_loss_ratio = gains.mean() / (abs(losses.mean()) + 1e-12) if len(losses) > 0 else np.inf

    # 换手率（从 costs 反推）
    if "costs" in records.columns:
        daily_costs = records[records["is_decision"]]["costs"].values \
            if "is_decision" in records.columns else records["costs"].values
        avg_daily_cost = daily_costs.mean()
    else:
        avg_daily_cost = 0.0

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "profit_loss_ratio": profit_loss_ratio,
        "avg_daily_cost": avg_daily_cost,
        "final_nav": nav[-1],
        "n_periods": n_periods,
    }


def summary_table(metrics: dict) -> str:
    """格式化为可读的摘要字符串"""
    if "error" in metrics:
        return f"Error: {metrics['error']}"

    lines = [
        "=" * 50,
        "  回测绩效摘要",
        "=" * 50,
        f"  总收益:       {metrics['total_return']:>10.2%}",
        f"  年化收益:     {metrics['annual_return']:>10.2%}",
        f"  年化波动:     {metrics['annual_volatility']:>10.2%}",
        f"  Sharpe:       {metrics['sharpe_ratio']:>10.2f}",
        f"  最大回撤:     {metrics['max_drawdown']:>10.2%}",
        f"  Calmar:       {metrics['calmar_ratio']:>10.2f}",
        f"  胜率:         {metrics['win_rate']:>10.2%}",
        f"  盈亏比:       {metrics['profit_loss_ratio']:>10.2f}",
        f"  最终净值:     {metrics['final_nav']:>10.2f}",
        "=" * 50,
    ]
    return "\n".join(lines)
