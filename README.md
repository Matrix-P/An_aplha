# AoAlpha v1.0

基于 Python 的端到端量化投研系统，覆盖因子挖掘、模型合成、仓位管理、风控约束和回测验证全链路。

## 架构

```
get_data ──→ factor ──→ models ──→ strategy ──→ portfolio ──→ backtest
 (数据)      (因子)     (模型)      (策略)       (仓位+风控)    (回测)
   │           │          │           │             │            │
  日/时/分   DSL+LLM   ICIR+Ridge  纯信号评分   持仓池+风控   统一时钟
            23个算子                            11个组件      流动性模拟
```

## 安装

```bash
pip install -r requirements.txt
```

## 配置

API 密钥放在 `config/` 目录下（已加入 `.gitignore`）：

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

## 快速开始

```python
from get_data import get_api
from factor import compute_factors_for_pool
from models import ICIRWeighted
from strategy import TopNStrategy
from risk import MaxPosition
from portfolio import SignalPortfolio
from backtest import BacktestConfig, BacktestEngine, compute_metrics, summary_table

# 1. 获取数据
pro = get_api()
factor_df = compute_factors_for_pool(pro, {
    "mom": "SMA($close,20)/SMA($close,60)-1",
    "vol": "Std($close,10)/Std($close,30)",
}, stock_pool=["600519.SH", "000858.SZ"], start_date="2024-01-01", end_date="2024-06-30")

ret_s = compute_factors_for_pool(pro,
    {"ret": "(Ref($close,-1)-$close)/$close"},
    stock_pool=["600519.SH", "000858.SZ"],
    start_date="2024-01-01", end_date="2024-06-30"
)["ret"]

# 2. 训练模型
model = ICIRWeighted()
model.fit(factor_df.loc[:"2024-03-31"], ret_s.loc[:"2024-03-31"])

# 3. 构建仓位（策略 + 风控）
pf = SignalPortfolio(
    TopNStrategy([model]),
    [MaxPosition(0.10)],
    mode="top_n", top_n=20
)

# 4. 回测
cfg = BacktestConfig("2024-04-01", "2024-06-28")
records = BacktestEngine(cfg, pf).run(factor_df, ret_series=ret_s)
print(summary_table(compute_metrics(records)))
```

## 模块

| 模块 | 职责 | 入口 |
|------|------|------|
| `get_data` | 数据获取 + Parquet 缓存 | `get_api()`, `fetch_daily_from_tushare()` |
| `factor` | 因子 DSL + 计算 + IC + LLM 挖掘 | `compile_expression()`, `compute_factors_for_pool()` |
| `models` | 多因子合成 | `ICIRWeighted`, `RidgeRegression`, `EqualWeight` |
| `strategy` | 纯信号评分 | `TopNStrategy`, `WeightedStrategy`, `QuantileStrategy` |
| `portfolio` | 仓位管理 + 风控 | `SignalPortfolio`, `RLPortfolio` |
| `risk` | 可插拔风控组件 | `MaxPosition`, `TrailingStop`, `Commission` 等 |
| `backtest` | 统一时钟回测 + 流动性模拟 | `BacktestEngine`, `BacktestConfig` |

## 文档

- [使用指南](docs/usage_guide.md)
- [数据获取](docs/get_data_module.md)
- [因子模块](docs/factor_module.md)
- [模型模块](docs/models_module.md)
- [策略模块](docs/strategy_module.md)
- [仓位管理](docs/portfolio_module.md)
- [风控模块](docs/risk_module.md)
- [回测模块](docs/backtest_module.md)

## License

MIT
