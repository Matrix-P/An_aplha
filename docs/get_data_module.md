# get_data/ 数据获取层技术文档

## 1. 设计理念

增量磁盘缓存策略：每个股票 × 频率独立缓存，拉取前检查已覆盖区间，只补缺失部分。

```
请求区间 [2024-01-01, 2024-06-30]
         │
         ├─ 读 {code}_daily_intervals.txt → 已覆盖 [2024-01-01, 2024-03-15]
         ├─ 计算缺失 → [2024-03-16, 2024-06-30]
         ├─ 调 Tushare API 补数据
         ├─ 合并已有 + 新增 → 写回 parquet
         └─ 更新 intervals.txt
```

---

## 2. 核心函数

### 2.1 `data_cache` — 通用缓存装饰器

```python
@data_cache(cache_dir="data/", freq="daily", time_col="trade_date", gap="1D")
def fetch_xxx(ts_code, start_date, end_date):
    ...
```

| 参数 | 说明 | 日线 | 分钟线 | 小时线 |
|------|------|------|--------|--------|
| `cache_dir` | 缓存目录 | `data/` | `data/` | `data/` |
| `freq` | 频率标签（文件命名） | `daily` | `1min` | `60min` |
| `time_col` | 时间列名 | `trade_date` | `trade_time` | `trade_time` |
| `gap` | 区间合并阈值 | `1D` | `1min` | `1h` |

**缓存文件：**

```
data/{ts_code}_{freq}_data.parquet       # OHLCV 数据
data/{ts_code}_{freq}_intervals.txt      # 已覆盖区间元数据
```

**行为：** 被装饰函数返回 `pd.DataFrame` 时必须有 `time_col` 列。空 DataFrame 表示该区间无数据，也会被标记为已覆盖。

### 2.2 `fetch_daily_from_tushare`

```python
df = fetch_daily_from_tushare(pro, "600519.SH", "2024-01-01", "2024-06-30")
```

**返回列：** `trade_date`, `open`, `high`, `low`, `close`, `vol`, `amount`

**缓存文件：** `data/600519.SH_daily_data.parquet`

### 2.3 `fetch_minute_from_tushare`

```python
df = fetch_minute_from_tushare(pro, "600519.SH", "2024-01-02", "2024-01-02")
```

**注意：** 需要 Tushare 积分 ≥ 2000，接口为 `pro.stk_mins(freq='1min')`。纯日期输入自动扩展为当日 00:00:00 ~ 23:59:59。

**返回列：** `trade_time`, `open`, `high`, `low`, `close`, `vol`, `amount`

**缓存文件：** `data/600519.SH_1min_data.parquet`

### 2.4 `fetch_hourly_from_tushare`

```python
df = fetch_hourly_from_tushare(pro, "600519.SH", "2024-01-02", "2024-01-05")
```

**实现：** 调用 `fetch_minute_from_tushare` 获取 1 分钟线，resample 为 60 分钟 OHLCV：

```
60min OHLCV = 1min 聚合:
  open   → first
  high   → max
  low    → min
  close  → last
  vol    → sum
  amount → sum
```

A 股一天约 4–6 根 60 分钟 K 线（取决于 resample 边界对齐）。

**缓存文件：** `data/600519.SH_60min_data.parquet`

### 2.5 `get_api`

```python
from get_data import get_api
pro = get_api()  # 从 config/tushare.env 加载 TUSHARE_TOKEN
```

---

## 3. 区间管理算法

### `_merge_intervals(intervals, gap)`

合并重叠或相邻（间距 ≤ gap）的区间。

```
输入: [("01-01","01-05"), ("01-06","01-10")], gap=1D
输出: [("01-01","01-10")]                      # 相邻合并

输入: [("01-01 09:30","10:00"), ("01-01 10:05","10:30")], gap=1min
输出: [("01-01 09:30","10:00"), ("01-01 10:05","10:30")]  # 间隔 >1min 不合并
```

### `_subtract_interval(full_start, full_end, excluded, gap)`

从完整区间中减去已覆盖部分，返回缺失区间。

```
full:     [01-01 =============== 01-10]
excluded: [01-01 == 01-03]  [01-06 == 01-10]
missing:            [01-04, 01-05]            # 中间缺口
```

---

## 4. 缓存生命周期

```
第 1 次请求 [2024-01-01, 2024-01-31]
  → 无缓存 → 拉取全量 → 写入 parquet + intervals

第 2 次请求 [2024-01-01, 2024-01-31]
  → 缓存覆盖 → 直接返回（0 次 API 调用）

第 3 次请求 [2024-01-01, 2024-02-15]
  → 覆盖 [01-01, 01-31] + 缺失 [02-01, 02-15]
  → 仅拉取 [02-01, 02-15] → 合并 → 更新缓存
```

---

## 5. 日内数据注意事项

**日期扩展：** 传入纯日期 `"2024-01-02"` 时，内部自动扩展为 `"2024-01-02 00:00:00"` ~ `"2024-01-02 23:59:59"`。非交易时段返回空 → 标记为已覆盖 → 后续不再重复请求。

**缓存复用：** 小时线通过 `fetch_hourly_from_tushare` 获取时会先查分钟线缓存。如果分钟线已缓存，小时线直接从缓存聚合，不调 API。

**区间格式：** 日内数据的 intervals.txt 使用 `YYYY-MM-DD HH:MM:SS` 格式，日线使用 `YYYY-MM-DD`。

---

## 6. 使用示例

```python
from get_data import get_api, fetch_daily_from_tushare, fetch_hourly_from_tushare

pro = get_api()

# 日线（已有 10 只股票的缓存）
df_daily = fetch_daily_from_tushare(
    pro, "600519.SH", "2024-01-01", "2024-06-30")

# 小时线（首次拉取 1 分钟 → 聚合 → 缓存）
df_hourly = fetch_hourly_from_tushare(
    pro, "600519.SH", "2024-01-02", "2024-01-05")

# 自定义缓存
from get_data import data_cache

@data_cache(cache_dir="my_data", freq="5min", time_col="trade_time", gap="5min")
def fetch_5min(ts_code, start_date, end_date):
    ...
```
