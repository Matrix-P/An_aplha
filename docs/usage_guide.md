# AnAlpha v1.0 使用指南

## 快速开始

```python
# 1. 数据
from get_data import get_api, fetch_daily_from_tushare
pro = get_api()
df = fetch_daily_from_tushare(pro, "600519.SH", "2024-01-01", "2024-06-30")

# 2. 因子
from factor import compute_factors_for_pool
factor_df = compute_factors_for_pool(pro, {
    "mom": "SMA($close,20)/SMA($close,60)-1",
    "vol": "Std($close,10)/Std($close,30)",
}, stock_pool=["600519.SH", "000858.SZ", ...],
   start_date="2024-01-01", end_date="2024-06-30")

ret_s = compute_factors_for_pool(pro,
    {"ret": "(Ref($close,-1)-$close)/$close"},
    stock_pool=[...], start_date="2024-01-01", end_date="2024-06-30"
)["ret"]

# 3. 模型
from models import ICIRWeighted
model = ICIRWeighted()
model.fit(factor_df.loc[:"2024-03-31"], ret_s.loc[:"2024-03-31"])

# 4. 策略
from strategy import TopNStrategy
strategy = TopNStrategy([model])

# 5. 风控 + 仓位管理
from risk import MaxPosition, MaxLeverage
from portfolio import SignalPortfolio
pf = SignalPortfolio(strategy, [MaxPosition(0.10), MaxLeverage(1.0)],
                     mode="top_n", top_n=20)

# 6. 回测
from backtest import BacktestConfig, BacktestEngine, compute_metrics, summary_table
cfg = BacktestConfig("2024-04-01", "2024-06-28",
                     decision_freq="daily", data_freq="daily",
                     initial_capital=1_000_000, liquidity_enabled=True)
engine = BacktestEngine(cfg, pf)
records = engine.run(factor_df, ret_series=ret_s)
print(summary_table(compute_metrics(records)))
```

---

## 配置 API 密钥

在 `config/` 目录下创建两个 `.env` 文件：

**`config/tushare.env`：**
```
name=tushare
TUSHARE_TOKEN=你的tushare_token
```

**`config/llm.env`：**
```
name=llm
DEEPSEEK_API_KEY=你的deepseek_api_key
```

---

## 模块速览

### 因子表达式语法

```python
from factor import compile_expression, compute_factors_for_pool

# 变量: $open $high $low $close $vol $amount
# 算子: SMA EMA Mean Std Sum Max Min Rank Ref Delta Corr Cov RSI CSRank ...
expr = "CSRank(Mean($close, 20)) - CSRank(Mean($close, 60))"
fn = compile_expression(expr)
result = fn(df)  # → pd.Series
```

### LLM 因子挖掘

```python
from factor import optimize_factors_with_llm, get_api_key_from_env

result = optimize_factors_with_llm(
    pro=pro, stock_pool=[...],
    start_date="2024-01-01", end_date="2024-06-30",
    api_key=get_api_key_from_env(),
    total_factors_needed=100, top_n=20,
)
# → {"top_20_factors": [...], "summary_stats": {...}}
```

### 因子评估

```python
from factor import calc_ic_series, calc_icir

ic_df = calc_ic_series(factor, ret_s, by_date=True)
metrics = calc_icir(ic_df)
# → {"ICIR": 6.35, "RankICIR": 6.17, "mean_IC": 0.04, ...}
```

### 自定义因子表达式

```python
# 动量
"Mean($close, 20) / (Mean($close, 60) + 1e-12) - 1"

# 波动率
"Std($close, 10) / (Std($close, 30) + 1e-12)"

# 量价背离
"-1 * Corr($vol, Delta($close,1)/(Ref($close,1)+1e-12), 20)"

# 截面排名差
"CSRank(Mean($close, 20)) - CSRank(Mean($close, 60))"

# RSI 信号
"(RSI($close, 14) - 50) / 50"
```

---

## 三种典型策略配置

### 配置 A：经典多因子日频

```python
model = ICIRWeighted()
model.fit(train_factors, train_ret)

pf = SignalPortfolio(
    TopNStrategy([model]),
    [MaxPosition(0.10), MaxLeverage(1.0)],
    mode="top_n", top_n=20,
)

cfg = BacktestConfig("2024-01-01", "2024-12-31",
                     decision_freq="daily", data_freq="daily")
```

### 配置 B：日线决策 + 日内风控监控

```python
pf = SignalPortfolio(
    TopNStrategy([model]),
    [MaxPosition(0.10), TrailingStop(0.05), MaxDrawdownBlowout(0.15)],
    mode="top_n", top_n=20,
)

cfg = BacktestConfig("2024-01-01", "2024-12-31",
                     decision_freq="daily", data_freq="60min",
                     risk_check_freq="60min")
```

### 配置 C：RL 强化学习

```python
pf = RLPortfolio(
    [model], [MaxPosition(0.20)],
    hidden_dim=64, n_epochs=20,
)
pf.train(train_dates, factor_df, ret_s)

cfg = BacktestConfig("2024-01-01", "2024-12-31",
                     decision_freq="daily", data_freq="daily")
```

---

## 多模型集成

```python
# 多个模型并行
m1 = ICIRWeighted()
m2 = RidgeRegression(alpha=0.5)
m1.fit(train_f, train_r)
m2.fit(train_f, train_r)

strategy = TopNStrategy([m1, m2])  # 策略自动等权平均多模型评分
pf = SignalPortfolio(strategy, [...], mode="top_n", top_n=20)
```

---

## 自定义模型

```python
from models import ModelBase

class MyModel(ModelBase):
    def fit(self, factor_df, forward_return):
        # 训练逻辑
        self._fitted = True

    def predict(self, factor_df):
        # 返回 z-score
        return cross_section_normalize(...)
```

---

## 指标解读

| 指标 | 合理范围（A股日频） | 说明 |
|------|-------------------|------|
| ICIR | > 1.0（可用），> 2.0（好） | 因子稳定性 |
| Sharpe | > 1.0（合格），> 2.0（优秀） | 风险调整收益 |
| MaxDD | < 20%（合格），< 10%（优秀） | 最大回撤 |
| Calmar | > 1.0（合格），> 2.0（优秀） | 收益回撤比 |
| 换手率 | < 50%/日（双边） | 过高则成本侵蚀收益 |
