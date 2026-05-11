# risk/ 风控层技术文档

## 1. 设计理念

风控层是一组**可插拔的组件**，每个组件只做一件事：接收权重向量 + 上下文 → 输出修正后的权重向量。

**被 `portfolio/` 层调用**，按序执行风控链：

```python
# 在 Portfolio._apply_risk_chain 中
for component in risk_components:
    weights = component.apply(weights, context)
```

风控组件不关心权重是怎么来的（策略信号还是 RL 直接产出），只做约束修正。

---

## 2. 基类 `RiskComponent`

```python
class RiskComponent(ABC):
    def __init__(self, name: str)
    
    @abstractmethod
    def apply(self, weights: pd.Series, context: dict) -> pd.Series:
        ...
```

**weights**：`pd.Series`，index = stock_code，values = weight（正=多，负=空）

**context**：`dict`，通用上下文，不同组件按需取用：

| 字段 | 类型 | 含义 |
|------|------|------|
| `prev_weights` | Series | 上期权重 |
| `portfolio_value` | float | 当前组合净值 |
| `initial_capital` | float | 初始资金 |
| `stock_returns` | Series | 各股累计收益 |
| `sectors` | Series | 行业分类 |
| `market_cap` | Series | 市值 |

---

## 3. 组件分类

### 3.1 仓位约束 (`risk.position`)

| 组件 | 参数 | 行为 |
|------|------|------|
| `MaxPosition` | `max_weight=0.10` | 单票 ≤ 上限，超额按比例再分配 |
| `MaxLeverage` | `max_leverage=1.0` | 总权重绝对值之和 ≤ 上限 |
| `MinPosition` | `min_stocks=5` | 持仓不足 min 只则空仓 |
| `EqualizeWeights` | — | 所有非零权重等权化 |

**MaxPosition 算法：**
```
1. 正/负权重分别处理
2. 归一化 → 找超额 → 截断 → 按比例再分配 → 重复直至收敛
```

**计算示例：**
```
输入: [0.15, 0.30, 0.05, 0.40, 0.10], max_weight=0.20
Step1: 超过 0.20 → [0.30, 0.40]
Step2: 截断 → [0.15, 0.20, 0.05, 0.20, 0.10], 超额=0.30
Step3: 再分配 → result
输出: [0.20, 0.20, 0.20, 0.20, 0.20] (全压到上限则等权)
```

### 3.2 止损止盈 (`risk.stoploss`)

| 组件 | 参数 | 触发条件 | 行为 |
|------|------|----------|------|
| `TrailingStop` | `max_drawdown=0.10` | 自峰值回撤 > 10% | 清空全部持仓 |
| `MaxDrawdownBlowout` | `max_drawdown=0.20` | 自初始回撤 > 20% | 清空并标记熔断 |
| `TakeProfit` | `target_return=0.20` | 单股收益 > 20% | 该股减半仓 |

**context 依赖：**
```
TrailingStop      → portfolio_value
MaxDrawdownBlowout → portfolio_value, initial_capital
TakeProfit        → stock_returns
```

### 3.3 暴露控制 (`risk.exposure`)

| 组件 | 参数 | 行为 |
|------|------|------|
| `NetExposureLimit` | `max_net_exposure=0.3` | 净敞口（多-空）≤ 上限 |
| `SectorNeutral` | — | 每个行业内权重均值为 0 |
| `SizeNeutral` | `n_bins=2` | 每个市值分组内权重均值为 0 |

**context 依赖：**
```
SectorNeutral → sectors
SizeNeutral   → market_cap
```

### 3.4 成本模型 (`risk.cost`)

| 组件 | 参数 | 行为 |
|------|------|------|
| `Commission` | `commission_rate=0.00025`, `stamp_tax=0.0005` | 换手成本超限则缩减 |
| `Slippage` | `max_weight_adjustment=0.02` | 单票调仓幅度超限则截断 |

**Commission.estimate_cost()** 估算成本而不修改权重：
```python
cost = turnover × commission_rate × 2 + sell_turnover × stamp_tax
```

---

## 4. 风控链执行示例

```python
from risk import MaxPosition, MaxLeverage, SectorNeutral, Commission
from portfolio import SignalPortfolio
from strategy import TopNStrategy

# 风控链在 Portfolio 内部执行
risk_chain = [
    MaxPosition(max_weight=0.10),      # 1. 单票不超 10%
    MaxLeverage(max_leverage=1.0),     # 2. 总杠杆不超 100%
    SectorNeutral(),                    # 3. 行业中性化
    Commission(max_turnover_cost_pct=0.005),  # 4. 限制换手成本
]

pf = SignalPortfolio(
    strategy=TopNStrategy([model]),
    risk_components=risk_chain,
    mode="top_n",
)

context = {"prev_weights": prev_w, "sectors": sectors}
weights = pf.step(date, factor_df, context)
# weights 已过完整风控链
```

---

## 5. 设计原则

1. **单一职责**：每个组件只做一件事，组合使用
2. **无状态（尽量）**：除了 TrailingStop（需要记录峰值），其他组件都是纯函数
3. **容错**：context 缺失时不崩溃，返回原权重或空仓
4. **顺序有意义**：先约束仓位，再中性化，最后算成本
