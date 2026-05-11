# strategy/ 策略层技术文档

## 1. 设计理念

策略层是**纯信号生成层**：接收模型列表，输出全市场股票 z-score 评分。不管仓位、不管风控、不管买卖——那些交给 `portfolio/` 层。

```
Models ──predict()──┐
Model_A ──→ pred_A ─┤
Model_B ──→ pred_B ─┼── Strategy.score_stocks() → z-score Series → Portfolio
Model_C ──→ pred_C ─┤
```

---

## 2. 基类 `StrategyBase`

```python
class StrategyBase(ABC):
    def __init__(self, models: List[ModelBase], name: str)

    @abstractmethod
    def score_stocks(self, date, factor_df) -> pd.Series:
        """返回全市场股票的 z-score 评分"""

    def _combine_predictions(self, factor_df) -> pd.Series:
        """多模型预测等权平均 → z-score"""
```

---

## 3. 三个策略

| 策略 | 职责 | 本质 |
|------|------|------|
| `TopNStrategy` | 多模型评分等权平均 | `_combine_predictions()` |
| `WeightedStrategy` | 同上 | `_combine_predictions()` |
| `QuantileStrategy` | 同上 | `_combine_predictions()` |

**注意：** 三个策略目前评分逻辑相同（等权平均），区分在于 Portfolio 层如何把评分转成权重（top_n/weighted/quantile）。策略名对应预期使用方式，方便配置可读性。

---

## 4. 与 Portfolio 的关系

```
Strategy.score_stocks(date, factor_df) → z-score Series (全市场)
        │
        ▼
Portfolio.step(date, factor_df, context)
        │
        ├─ 用 z-score 更新持仓池（入池/保留/踢出）
        ├─ 评分 → 原始权重（top_n / weighted / quantile）
        ├─ 过风控链
        └─ 返回 T+1 目标权重
```

---

## 5. 自定义策略

```python
class MyStrategy(StrategyBase):
    def score_stocks(self, date, factor_df):
        cross = factor_df.xs(date, level=0)
        # 自定义评分逻辑
        scores = ...
        return (scores - scores.mean()) / (scores.std() + 1e-12)
```
