"""回测配置"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BacktestConfig:
    """回测频率和参数配置

    decision_freq / data_freq / risk_check_freq:
        'daily' | '60min' | '30min' | '1min' | None（与 decision 同步）
    """

    start_date: str                     # "2024-01-01"
    end_date: str                       # "2024-06-30"

    decision_freq: str = "daily"        # 决策频率
    data_freq: str = "daily"            # 数据粒度
    risk_check_freq: Optional[str] = None  # 风控巡检频率，None=与决策同步

    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.00025    # 万2.5
    stamp_tax: float = 0.0005           # 卖出印花税 0.05%

    # 流动性模拟
    liquidity_enabled: bool = True
    # 每 tick 最多成交的股数 ~ N(volume/k, volume/(2k)), 截断到 [0, volume]
    liquidity_mean_frac: float = 0.5    # volume * mean_frac 为中心
    liquidity_std_frac: float = 0.25    # volume * std_frac 为标准差

    def __post_init__(self):
        # 默认：风控频率 = 决策频率
        if self.risk_check_freq is None:
            self.risk_check_freq = self.decision_freq

    @property
    def freq_rank(self):
        """频率粒度排序，用于确定最细粒度"""
        _map = {"daily": 0, "60min": 1, "30min": 2, "1min": 3}
        return max(_map.get(f, 0) for f in
                   [self.decision_freq, self.data_freq, self.risk_check_freq])
