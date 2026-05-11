# backtest/ 回测模块技术文档

## 1. 设计理念

统一时钟回测引擎：按最细频率生成时间轴，在决策点调 portfolio，在所有 tick 执行订单和风控。

```
BacktestConfig ──→ Clock(时间轴) ──→ Engine(主循环) ──→ Performance(指标)
                      │                   │
                      │    ┌──────────────┼──────────────┐
                      │    ▼              ▼              ▼
                      │  Decision Tick  Risk Tick    Data Tick
                      │  portfolio      止损巡检      更新净值
                      │  .step()
                      │    │
                      ▼    ▼
                   Broker.execute()
                   (流动性约束)
```

---

## 2. 配置 `BacktestConfig`

```python
from backtest import BacktestConfig

cfg = BacktestConfig(
    start_date="2024-01-01",
    end_date="2024-06-30",

    # 频率配置
    decision_freq="daily",       # 决策频率: daily / 60min / 30min / 1min
    data_freq="daily",           # 数据粒度
    risk_check_freq=None,        # 风控巡检频率，None=与决策同步

    # 资金与成本
    initial_capital=1_000_000,
    commission_rate=0.00025,     # 万2.5
    stamp_tax=0.0005,            # 卖出印花税

    # 流动性模拟
    liquidity_enabled=True,
    liquidity_mean_frac=0.5,     # 成交中心 = volume * 0.5
    liquidity_std_frac=0.25,     # 标准差 = volume * 0.25
)
```

### 三种典型配置

| 场景 | decision_freq | data_freq | risk_check_freq | 说明 |
|------|--------------|-----------|-----------------|------|
| 纯日线 | daily | daily | None | 经典日频回测，每天调仓 |
| 日线决策+日内风控 | daily | 60min | 60min | 每小时检查止损，用到小时线 |
| 未来：全日内 | 60min | 60min | 60min | 模型/策略全在小时级跑 |

---

## 3. 统一时钟 `generate_timeline`

按 `data_freq` 的最细粒度生成时间轴，标注每个 tick 的角色。

```python
from backtest.clock import generate_timeline

tl = generate_timeline(cfg)
# → DataFrame:
#   time                is_decision  is_risk_check  is_data  date_label
#   2024-01-02 00:00:00     True        True         True    2024-01-02
#   2024-01-03 00:00:00     True        True         True    2024-01-03
#   ...
```

**标注规则：**
- `is_decision`: 每天第一个 tick（daily）、或每个 tick（日内频率）
- `is_risk_check`: 同上
- `is_data`: 始终为 True

**日内数据自动过滤 A 股交易时段**（9:30-11:30, 13:00-15:00）。

---

## 4. Broker 订单执行

### 执行流程

```
target_weights ──→ 计算调仓差额 ──→ 流动性约束 ──→ 实际成交权重
       │                                                │
       ▼                                                ▼
  佣金 + 印花税                                      成交率
```

### 成交量流动性模型

```
max_fillable_shares ~ N(volume × mean_frac, volume × std_frac)
                            截断到 [0, volume]

如果 |目标调仓| > max_fillable → 仅成交 max_fillable 部分
如果 volume = 0            → 无法成交
```

**示例：** 某股票 tick 成交量 100 万股，均值 0.5 标准差 0.25：
- 大约 68% 概率可成交 25~75 万股
- 大约 95% 概率可成交 0~100 万股
- 几乎不会超过成交量本身

```python
from backtest import Broker

broker = Broker(liquidity_enabled=True, seed=42)

result = broker.execute(
    target_weights=target,       # Series, 目标权重
    current_positions=current,   # Series, 当前权重
    prices=prices,               # Series, 当前价格
    volumes=volumes,             # Series, 成交量（股）
    portfolio_value=1_000_000,   # 组合净值
)
# → {
#     "filled_weights": ...   # 实际成交权重
#     "costs": ...            # 总交易成本
#     "fill_rates": ...       # 每只股票成交率
#     "unfilled": ...         # 未成交差额
# }
```

---

## 5. 引擎 `BacktestEngine`

### 主循环

```python
from backtest import BacktestEngine

engine = BacktestEngine(cfg, portfolio)
records = engine.run(factor_df, ret_series=ret_s)

# records 每 tick 一行:
#   time, date, is_decision, is_risk_check,
#   nav, period_return, costs, positions, max_drawdown
```

**每个 tick 的逻辑：**

```
if is_decision:
    portfolio.step(date, factor_df, context) → target_weights

if is_risk_check:
    止损组件检查 → 必要时清仓

Broker.execute(target_weights, current_positions, prices, volumes)
    → 实际成交权重 + 成本

NAV *= (1 + period_return) - costs
记录 nav, positions, drawdown, ...
```

### 风控巡检

引擎区分风控组件类型：止损类（`TrailingStop`, `MaxDrawdownBlowout`）在每个 risk_check tick 都会执行，仓位约束类只在 decision tick 通过 portfolio 执行。

### 数据获取

引擎从 `factor_df` (MultiIndex) 中按日期切片获取截面数据传给 portfolio。价格和成交量从可选 DataFrame 中同样按 tick 切片。

---

## 6. 绩效指标 `compute_metrics`

```python
from backtest import compute_metrics, summary_table

metrics = compute_metrics(records, periods_per_year=252)
print(summary_table(metrics))
```

**输出指标：**

| 指标 | 计算 |
|------|------|
| `total_return` | (final_nav / init_nav) - 1 |
| `annual_return` | (1 + total_return)^(252/n) - 1 |
| `annual_volatility` | std(daily_returns) × √252 |
| `sharpe_ratio` | (annual_return - 0.02) / annual_vol |
| `max_drawdown` | max((peak - nav) / peak) |
| `calmar_ratio` | annual_return / max_drawdown |
| `win_rate` | 正收益周期占比 |
| `profit_loss_ratio` | 平均盈利 / 平均亏损 |
| `avg_daily_cost` | 日均交易成本 |

---

## 7. 完整使用示例

```python
from backtest import BacktestConfig, BacktestEngine, compute_metrics, summary_table
from portfolio import SignalPortfolio
from strategy import TopNStrategy
from models import ICIRWeighted
from risk import MaxPosition
from get_data import get_api

# 1. 训练模型
pro = get_api()
model = ICIRWeighted()
model.fit(train_factors, train_ret)

# 2. 构建策略和 portfolio
strategy = TopNStrategy([model])
pf = SignalPortfolio(strategy, [MaxPosition(0.10)], mode="top_n", top_n=20)

# 3. 配置回测
cfg = BacktestConfig(
    start_date="2024-01-02",
    end_date="2024-06-28",
    decision_freq="daily",
    data_freq="daily",
    initial_capital=1_000_000,
    liquidity_enabled=True,
)

# 4. 运行
engine = BacktestEngine(cfg, pf)
records = engine.run(factor_df, ret_series=ret_s)

# 5. 绩效
metrics = compute_metrics(records)
print(summary_table(metrics))
```

---

## 8. 设计原则

1. **频率无感**：通过配置切换频率，不重写代码
2. **流动性真实**：成交量约束让大资金回测结果更可信
3. **风控真实**：止损在日内 tick 级巡检，不等到调仓日
4. **可扩展**：加新频率只需在 `FREQ_MAP` 添加映射
