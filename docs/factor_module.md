# factor/ 因子引擎技术文档

## 1. 架构概览

```
表达式字符串 "SMA($close, 20) / SMA($close, 60) - 1"
        │
        ▼
factor_compiler.py          ──→ 可调用函数 calc_factor(df) → Series
        │
        ▼
factor_compute.py           ──→ 多股票批量计算 → MultiIndex DataFrame
        │
        ▼
ic_compute.py               ──→ IC / RankIC / ICIR 评估
        │
        ▼
factor_llm.py               ──→ LangGraph LLM 自动挖掘因子
```

---

## 2. factor_compiler.py — 因子表达式 DSL

### 2.1 算子注册

全局字典 `_operators`，通过装饰器注册：

```python
@register_operator("SMA")
def _sma(series, window):
    window = int(window)
    return series.rolling(window, min_periods=1).mean()
```

### 2.2 已注册算子（23个）

| 类别 | 算子 | 签名 | 含义 |
|------|------|------|------|
| 基础算术 | +, -, *, / | 隐式 | 四则运算 |
| 一元函数 | `Abs`, `Log`, `Sign`, `Sqrt` | series → series | 数学变换 |
| 时间序列 | `Ref(x, n)` | (series, int) → series | n 期前值 |
| | `Delta(x, n)` | (series, int) → series | n 日变化 |
| | `Mean(x, n)` | (series, int) → series | 滚动均值 |
| | `Std(x, n)` | (series, int) → series | 滚动标准差 |
| | `Sum(x, n)` | (series, int) → series | 滚动求和 |
| | `Max(x, n)` | (series, int) → series | 滚动最大值 |
| | `Min(x, n)` | (series, int) → series | 滚动最小值 |
| | `Rank(x, n)` | (series, int) → series | 时序百分比排名 |
| 均线 | `SMA(x, n)` / `MA(x, n)` | (series, int) → series | 简单移动平均 |
| | `EMA(x, n)` | (series, int) → series | 指数移动平均 |
| 指标 | `RSI(x, n)` | (series, int) → series | 相对强弱指数 |
| 截面 | `CSRank(x)` | (series) → series | 截面排名(0~1) |
| | `CSMean(x)` | (series) → series | 截面均值 |
| | `CSStd(x)` | (series) → series | 截面标准差 |
| 相关 | `Corr(x, y, n)` | (series, series, int) → series | 滚动相关系数 |
| | `Cov(x, y, n)` | (series, series, int) → series | 滚动协方差 |
| 缩放 | `Scale(x, a)` | (series, float) → series | 线性缩放 |
| | `Clip(x, l, u)` | (series, float, float) → series | 截断 |

### 2.3 表达式语法

```
变量:   $close, $open, $high, $low, $vol, $amount
常量:   1, -0.5, 100
函数:   FuncName(arg1, arg2, ...)
算术:   + - * /
括号:   ( )
```

**表达式示例：**

```python
# 动量因子
"SMA($close, 20) / (SMA($close, 60) + 1e-12) - 1"

# 波动率因子
"Std($close, 10) / (Std($close, 30) + 1e-12)"

# 量价背离
"-1 * Corr($vol, ($close - Ref($close, 1)) / (Ref($close, 1) + 1e-12), 20)"

# 截面排名差
"CSRank(Mean($close, 20)) - CSRank(Mean($close, 60))"
```

### 2.4 编译流程

```
Tokenize → Parse (递归下降) → AST → Generate Python Code → exec() → callable
```

核心函数：`compile_expression(expr_str, df_var='df') -> callable`

返回函数签名：`calc_factor(df: pd.DataFrame) -> pd.Series`

---

## 3. factor_compute.py — 批量因子计算

### 3.1 核心函数

```python
compute_factors_for_pool(
    pro,                    # Tushare API 对象
    factor_dict,            # {"因子名": "表达式"}
    stock_pool,             # ["600519.SH", "000858.SZ", ...]
    start_date,             # "2024-01-01"
    end_date=None,          # "2024-12-31"
    lookback_days=None,     # 回溯天数（None则自动计算）
    auto_lookback=True,     # 自动从表达式推断回溯窗口
) -> pd.DataFrame           # MultiIndex (trade_date, stock), columns = 因子名
```

### 3.2 自动回溯窗口

`extract_max_lookback(expr)` 从表达式提取最大窗口大小，如 `SMA($close, 60)` → 60。`get_max_lookback_from_exprs` 取所有因子窗口的 2 倍（安全缓冲）。

### 3.3 计算流程

```
股票池 → 逐只获取数据(含回溯) → 逐只计算因子 → concat → MultiIndex
```

---

## 4. ic_compute.py — IC 评估

### 4.1 IC 计算

```python
calc_ic_series(
    factor,              # MultiIndex (date, stock) Series
    forward_ret,         # MultiIndex (date, stock) Series
    by_date=True,        # True=横截面, False=时间序列
    trim_quantile=None,  # (0.01, 0.99) 截尾
    min_samples=3,       # 每日最少样本
) -> pd.DataFrame        # columns: [trade_date, IC, RankIC]
```

**计算过程：**
- 横截面模式：按日期分组，每日期内做 Pearson 和 Spearman 相关
- 时间序列模式：整个序列直接做相关

### 4.2 ICIR 计算

```python
calc_icir(ic_df) -> dict
# {
#     "mean_IC": 0.04,       # 平均 IC
#     "std_IC": 0.10,        # IC 标准差
#     "ICIR": 6.35,          # mean/std * sqrt(252)
#     "mean_RankIC": 0.035,
#     "std_RankIC": 0.09,
#     "RankICIR": 6.17,
# }
```

---

## 5. factor_llm.py — LLM 因子挖掘

### 5.1 IC 综合评分

```
IC综合 = 0.5 × |mean_IC| × |ICIR| + 0.5 × |mean_RankIC| × |RankICIR|
```

为何不用纯 ICIR：低均值低波动的因子（如 mean=0.002, std=0.0005 → ICIR=31.7）会在纯 ICIR 下虚高。`|mean| × |ICIR|` 惩罚了低均值场景。

### 5.2 FactorOptimizer 类

```python
optimizer = FactorOptimizer(
    pro, stock_pool, start_date, end_date,
    api_key, llm_model_uri, llm_model,
    batch_size=20, total_factors_needed=100, top_n=20,
)

result = optimizer.run_optimization()
# → {"top_20_factors": [...], "summary_stats": {...}}
```

### 5.3 LangGraph 工作流

```
start → generate_factors ──→ evaluate_factors ──→ collect_results
           ↑                        │                    │
           └────────────────────────┘                    │
                                          (不够继续生成)   │
                                                         ▼
                                                        end
```

LLM 被要求生成有金融意义的因子表达式，每批 20 个，计算后评估 IC 综合，已收集的因子信息会反馈给 LLM 避免重复。

### 5.4 便捷函数

```python
optimize_factors_with_llm(pro, stock_pool, ...) -> dict  # 一键运行
compute_return_factor(pro, stock_pool, ...) -> Series     # 计算 t+1 收益
get_api_key_from_env("DEEPSEEK_API_KEY") -> str           # 从 config/llm.env 读密钥
```
