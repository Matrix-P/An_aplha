# models/ 模型层技术文档

## 1. 设计理念

模型层的职责：**将多因子合成单一预测信号**。

每个模型是一个独立类，接收因子 DataFrame → 产出预测 Series。多个模型可以并行运行，各自独立预测，供策略层统合调度。

```
因子 DataFrame ──┬──→ Model A.predict() ──→ pred_A ──┐
    (date,stock) ├──→ Model B.predict() ──→ pred_B ──┼──→ Strategy
                 └──→ Model C.predict() ──→ pred_C ──┘
```

---

## 2. 接口约定

### 2.1 数据格式

```python
# 输入
factor_df: pd.DataFrame
# 索引: MultiIndex (trade_date, stock)
# 列:   factor_strong, factor_mid, factor_weak, ...

# 训练标签
forward_return: pd.Series
# 索引: MultiIndex (trade_date, stock)
# 值:   t+1 日收益率

# 输出
prediction: pd.Series
# 索引: MultiIndex (trade_date, stock)
# 值:   z-score（截面标准化后的预测值）
```

### 2.2 基类 `ModelBase`

```python
class ModelBase(ABC):
    def __init__(self, name: str = "model")
    
    @abstractmethod
    def fit(self, factor_df, forward_return) -> None:
        """用历史数据训练模型"""
    
    @abstractmethod
    def predict(self, factor_df) -> pd.Series:
        """产出预测信号（z-score）"""
    
    @property
    def is_fitted(self) -> bool
    
    def save(self, path) -> None       # pickle 序列化
    @staticmethod
    def load(path) -> ModelBase        # pickle 反序列化
```

### 2.3 截面标准化工具

```python
from models.base import cross_section_normalize

normed = cross_section_normalize(factor_df)
# 每个日期截面的每列因子 → (x - mean) / (std + 1e-12)
```

---

## 3. 三个模型详解

### 3.1 EqualWeight — 等权合成

**原理：** 所有因子截面标准化后取平均值。

```
prediction = mean(zscore(factor_1), zscore(factor_2), ...)
```

**特点：**
- 无需训练（`fit` 是空操作）
- 最简单可靠的基线模型
- 不区分因子好坏，所有因子一视同仁

**使用：**
```python
model = EqualWeight()
model.fit(factor_df, ret)   # 空操作，但保持接口一致
pred = model.predict(factor_df)
```

---

### 3.2 ICIRWeighted — ICIR 加权合成

**原理：** 在训练期计算每个因子的 ICIR，以其绝对值作为权重，加权合成。

```
prediction = Σ (|ICIR_i| × zscore(factor_i)) / (Σ |ICIR_i| + 1e-12)
```

**训练过程：**
1. 对每个因子，调用 `calc_ic_series` 计算训练期内的横截面 IC 序列
2. 调用 `calc_icir` 计算年化 ICIR
3. 权重 = `abs(ICIR)`

**特点：**
- 好因子权重高，差因子权重低
- 如果某个因子在训练期表现差（ICIR 接近 0），几乎不会被纳入
- 依赖历史 IC 稳定性假设

**初始化和使用：**
```python
model = ICIRWeighted(trim_quantile=(0.01, 0.99), min_samples=5)
model.fit(train_factors, train_ret)  # 计算各因子权重
pred = model.predict(factor_df)      # 加权合成

print(model.weights)                 # {'factor_A': 6.35, 'factor_B': 3.17, ...}
```

---

### 3.3 RidgeRegression — Ridge 回归

**原理：** 用截面标准化后的因子值做 Ridge 回归（L2 正则化），拟合下期收益率。

```
prediction = Σ (β_i × zscore(factor_i))
其中 β = argmin ||Factor × β - ret||² + α||β||²
```

**训练过程：**
1. 所有因子截面标准化
2. 与收益对齐，剔除 NaN
3. 用 `sklearn.linear_model.Ridge` 拟合（`fit_intercept=False`）
4. 如果样本太少（< 因子数 + 5），自动退化为等权

**特点：**
- 能学习因子间的最优线性组合权重
- 允许因子间有负相关（对冲效应）
- L2 正则化防止过拟合
- 小样本下安全退化为等权

**初始化和使用：**
```python
model = RidgeRegression(alpha=1.0)   # alpha=正则化强度
model.fit(train_factors, train_ret)
pred = model.predict(factor_df)

# 查看学到的系数
print(model.model.coef_)             # [β_1, β_2, ...]
```

---

## 4. 模型评估

### 4.1 `evaluate_model` — 单模型评估

```python
from models.evaluation import evaluate_model

result = evaluate_model(
    prediction=pred,           # 模型预测值
    forward_return=ret,        # 实际下期收益
    trim_quantile=(0.01, 0.99),
    min_samples=5,
    n_groups=5,                # 分组收益组数
)
```

**返回字典：**

| 键 | 类型 | 含义 |
|----|------|------|
| `mean_IC` | float | 平均 Pearson IC |
| `std_IC` | float | IC 标准差 |
| `ICIR` | float | 年化 IC Information Ratio |
| `mean_RankIC` | float | 平均 Spearman Rank IC |
| `RankICIR` | float | 年化 Rank ICIR |
| `turnover` | float | 双边换手率 |
| `group_returns` | DataFrame | 按预测值分组的各组平均收益 |
| `ic_df` | DataFrame | 每日 IC 序列 |

### 4.2 `compare_models` — 多模型对比

```python
from models.evaluation import compare_models

cmp = compare_models(
    predictions={"等权": pred_a, "ICIR加权": pred_b},
    forward_return=ret,
)
# → DataFrame 含 ICIR, RankICIR, mean_IC, 换手率
```

---

## 5. 使用示例

```python
from models import EqualWeight, ICIRWeighted, RidgeRegression
from models.evaluation import evaluate_model, compare_models
from get_data import get_api, fetch_daily_from_tushare
from factor import compute_factors_for_pool

# 1. 准备数据
pro = get_api()
factors = compute_factors_for_pool(pro, {
    "mom": "SMA($close,20)/SMA($close,60)-1",
    "vol": "Std($close,10)/Std($close,30)",
    "liq": "Log(1+$vol)",
}, stock_pool=["600519.SH", ...], start_date="2024-01-01")

ret = compute_factors_for_pool(pro,
    {"ret": "(Ref($close,-1)-$close)/$close"},
    stock_pool=[...], start_date="2024-01-01"
)["ret"]

# 2. 训练期/预测期划分
train_f = factors.loc[:"2024-06-30"]
train_r = ret.loc[:"2024-06-30"]

# 3. 训练并预测
models = [
    EqualWeight(),
    ICIRWeighted(),
    RidgeRegression(alpha=1.0),
]

for m in models:
    m.fit(train_f, train_r)

predictions = {m.name: m.predict(factors) for m in models}

# 4. 比较
print(compare_models(predictions, ret))
```

---

## 6. 设计原则

1. **统一接口**：所有模型遵循 `fit/predict` 接口，策略层无需知道模型内部细节
2. **截面标准化**：所有模型内部先做截面 z-score，避免量纲差异
3. **防御性编程**：Ridge 小样本退化为等权；ICIR 权重对未覆盖因子置零
4. **可序列化**：支持 pickle save/load，便于生产部署
