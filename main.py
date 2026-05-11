"""AnAlpha v1.0 — 全链路演示"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from get_data import get_api, fetch_daily_from_tushare
from factor import compute_factors_for_pool, calc_ic_series, calc_icir
from models import ICIRWeighted, RidgeRegression, EqualWeight, compare_models
from strategy import TopNStrategy
from risk import MaxPosition, MaxLeverage, MaxDrawdownBlowout
from portfolio import SignalPortfolio
from backtest import BacktestConfig, BacktestEngine, compute_metrics, summary_table

# ── 1. 数据准备 ──
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

pro = get_api()
STOCK_POOL = [
    "600519.SH", "000858.SZ", "300750.SZ", "002475.SZ", "600941.SH",
    "002415.SZ", "300059.SZ", "000333.SZ", "600276.SH", "300124.SZ",
]
START, END = "2024-01-01", "2025-12-31"

# 因子
print(f"  计算因子（{len(STOCK_POOL)} 只股票）...")
factor_df = compute_factors_for_pool(pro, {
    "mom_20": "Mean($close, 20) / (Ref($close, 20) + 1e-12) - 1",
    "mom_60": "Mean($close, 60) / (Ref($close, 60) + 1e-12) - 1",
    "volatility": "Std($close, 20) / ($close + 1e-12)",
    "volume_ratio": "Mean($vol, 5) / (Mean($vol, 20) + 1e-12) - 1",
    "price_reversal": "-1 * ($close - Ref($close, 5)) / (Ref($close, 5) + 1e-12)",
    "close_position": "($close - $low) / ($high - $low + 1e-12)",
    "log_volume": "Log(1 + $vol)",
    "delta_vol": "Delta($vol, 5) / (Mean($vol, 5) + 1e-12)",
}, stock_pool=STOCK_POOL, start_date=START, end_date=END)

# 下期收益
print("  计算下期收益...")
ret_s = compute_factors_for_pool(pro,
    {"ret_t1": "(Ref($close, -1) - $close) / ($close + 1e-12)"},
    stock_pool=STOCK_POOL, start_date=START, end_date=END,
)["ret_t1"]

print(f"  因子: {factor_df.shape}, 收益: {ret_s.shape}")
print(f"  日期范围: {factor_df.index.get_level_values(0).min()} ~ {factor_df.index.get_level_values(0).max()}")

# ── 2. 因子评估 ──
print("\n" + "=" * 60)
print("2. 因子评估")
print("=" * 60)

for col in factor_df.columns:
    ic_df = calc_ic_series(factor_df[col], ret_s, by_date=True)
    m = calc_icir(ic_df)
    comp = 0.5 * abs(m['mean_IC']) * abs(m['ICIR']) + \
           0.5 * abs(m['mean_RankIC']) * abs(m['RankICIR'])
    print(f"  {col:<20} ICIR={m['ICIR']:>7.2f}  RankICIR={m['RankICIR']:>7.2f}  IC综合={comp:>7.4f}")

# ── 3. 训练模型 ──
print("\n" + "=" * 60)
print("3. 训练模型")
print("=" * 60)

dates = sorted(factor_df.index.get_level_values(0).unique())
split_idx = int(len(dates) * 0.7)
train_end = dates[split_idx]

train_f = factor_df.loc[:train_end]
train_r = ret_s.loc[:train_end]

m_icir = ICIRWeighted()
m_ridge = RidgeRegression(alpha=1.0)
m_ew = EqualWeight()

m_icir.fit(train_f, train_r)
m_ridge.fit(train_f, train_r)
m_ew.fit(train_f, train_r)

print(f"  ICIRWeighted weights: {m_icir.weights}")
print(f"  Ridge coef: {m_ridge.model.coef_ if m_ridge.model else '退化为等权'}")
print(f"  训练期: {train_f.index.get_level_values(0).min()} ~ {train_end}")
print(f"  测试期: {dates[split_idx]} ~ {dates[-1]}")

# ── 4. 模型评估 ──
print("\n" + "=" * 60)
print("4. 模型评估（全样本）")
print("=" * 60)

preds = {
    "ICIR加权": m_icir.predict(factor_df),
    "Ridge": m_ridge.predict(factor_df),
    "等权": m_ew.predict(factor_df),
}
cmp = compare_models(preds, ret_s)
print(cmp.to_string())

# ── 5. 回测 ──
print("\n" + "=" * 60)
print("5. 回测")
print("=" * 60)

# 选最好的模型做回测（Ridge 样本外 IC 更好）
best_model = m_ridge

pf = SignalPortfolio(
    strategy=TopNStrategy([best_model]),
    risk_components=[MaxPosition(0.20), MaxLeverage(1.0)],
    mode="top_n", top_n=3,
    entry_threshold=-99.0,  # 低门槛让持仓池初始化
)

cfg = BacktestConfig(
    start_date=train_end,
    end_date=dates[-1],
    decision_freq="daily",
    data_freq="daily",
    initial_capital=1_000_000,
    liquidity_enabled=True,
)

engine = BacktestEngine(cfg, pf)
records = engine.run(factor_df, ret_series=ret_s)

# 绩效
decisions = records[records["is_decision"]]
metrics = compute_metrics(records)
print(summary_table(metrics))

# 净值曲线概况
if len(decisions) > 0:
    nav = decisions["nav"].values
    print(f"\n  回测期: {train_end} ~ {dates[-1]}")
    print(f"  交易日数: {len(decisions)}")
    print(f"  累计收益: {(nav[-1]/nav[0]-1)*100:.2f}%")
    print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")

print("\n" + "=" * 60)
print("演示完毕")
print("=" * 60)
