# portfolio/ 仓位管理层技术文档

## 1. 设计理念

Portfolio 是**仓位管理和风控执行的统一层**。策略只管"哪只股票好"，Portfolio 管"买什么、买多少、能不能买、什么时候卖"。

**T 日闭盘后执行：**

```
T 日数据就绪
    │
    ▼
Portfolio.step(date, factor_df, context)
    │
    ├── 1. 获取评分（来自 Strategy 或 RL Agent）
    ├── 2. 更新持仓池（入池 / 保留 / 踢出）
    ├── 3. 评分 → 原始权重
    ├── 4. 过风控链: component_1 → component_2 → ...
    ├── 5. 清理零仓位 → 移出持仓池
    └── 6. 返回 T+1 目标权重
```

**持仓池规则：**
- 池内有持仓 → 保留
- 池内有持仓 + 信号差 → 目标权重降为 0，闭盘后踢出
- 池外无持仓 + 信号强 → 加入池

---

## 2. 两种 Portfolio

### 2.1 SignalPortfolio — 依赖策略

需要传入一个 `Strategy`，由策略产生评分，Portfolio 负责转换。

```python
from portfolio import SignalPortfolio
from strategy import TopNStrategy
from risk import MaxPosition, MaxLeverage

pf = SignalPortfolio(
    strategy=TopNStrategy([model]),
    risk_components=[MaxPosition(0.10), MaxLeverage(1.0)],
    mode="top_n",           # top_n | weighted | quantile
    top_n=20,
    entry_threshold=0.5,    # 入池 z-score 阈值
    exit_threshold=-1.0,    # 踢出阈值
)

# 每日执行
weights = pf.step("2024-06-01", factor_df, context)
```

**三种分配模式：**

| mode | 逻辑 |
|------|------|
| `top_n` | 池内选评分最高的 N 只，等权 |
| `weighted` | 池内正分做多（比例化）、负分做空 |
| `quantile` | 池内 top_q 做多、bottom_q 做空 |

### 2.2 RLPortfolio — 独立训练

不需要 Strategy。Agent 直接从因子数据学习最优权重分配。风控内嵌在训练环境中。

```python
from portfolio import RLPortfolio

pf = RLPortfolio(
    models=[model],
    risk_components=[MaxPosition(0.20)],
    hidden_dim=64,
    n_epochs=20,
    lr=3e-4,
    temperature=1.0,
    entropy_coef=0.01,
    feature_cols=["mom", "vol"],
)

# 训练
metrics = pf.train(dates, factor_df, ret_series)

# 推理
weights = pf.step("2024-06-01", factor_df, context)
```

**网络结构：**
```
PortfolioNet(feature_dim → 64 → 64 → 1)
每只股票独立通过共享 MLP → 全股票 softmax → 权重
```

**训练算法：Actor-Critic (REINFORCE + Value baseline)**
- Policy Loss: `-log_prob × advantage`
- Value Loss: MSE(value, return)
- Entropy Bonus: 防止权重过度集中

---

## 3. 基类 `PortfolioBase`

```python
class PortfolioBase(ABC):
    def __init__(self, risk_components, entry_threshold, exit_threshold, name):
        self.risk_components = risk_components  # 风控链
        self.pool = set()       # 持仓候选池
        self.positions = {}     # 当前持仓权重

    @abstractmethod
    def step(self, date, factor_df, context) -> pd.Series:
        ...

    def _apply_risk_chain(self, weights, context) -> pd.Series:
        for comp in self.risk_components:
            weights = comp.apply(weights, context)
        return weights

    def _update_pool(self, scores):
        """根据评分更新持仓池（入池/保留/踢出）"""

    def _cleanup_pool(self, weights):
        """零仓位移出持仓池"""
```

---

## 4. 与 Strategy / Risk 的关系

```
Strategy               Portfolio                    Risk
─────────              ─────────                    ────
score_stocks()  ──→   评分转权重                   
                       持仓池管理                   
                       风控链执行  ──────────────→  apply(weights, ctx)
                       返回最终权重
```

| 场景 | Portfolio | Strategy | 说明 |
|------|-----------|----------|------|
| 传统多因子 | SignalPortfolio | 需要 | 策略评分 → Portfolio 分配 |
| 强化学习 | RLPortfolio | 不需要 | Agent 直接学权重，不通过 Strategy |

---

## 5. 扩展自定义 Portfolio

```python
class MyPortfolio(PortfolioBase):
    def step(self, date, factor_df, context):
        scores = ...  # 自定义逻辑
        self._update_pool(scores)
        weights = self._scores_to_weights(scores)
        weights = self._apply_risk_chain(weights, context)
        self._cleanup_pool(weights)
        return weights
```
