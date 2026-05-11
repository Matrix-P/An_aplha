"""RL Portfolio：独立训练，不依赖 Strategy，直接产出权重"""
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Optional, List

from portfolio.base import PortfolioBase
from portfolio.rl_env import RLEnvironment
from portfolio.rl_agent import RLAgent
from risk.base import RiskComponent


class RLPortfolio(PortfolioBase):
    """基于 RL 的仓位管理器

    不需要 Strategy —— Agent 直接从因子数据学习最优权重分配。

    Parameters
    ----------
    models : list
        因子模型（生成预测作为特征）
    risk_components : list
        风控组件
    hidden_dim : int
        RL 网络隐层维度
    lr : float
        学习率
    n_epochs : int
        训练轮数
    temperature : float
        softmax 温度
    entropy_coef : float
        熵正则系数
    feature_cols : list, optional
        额外因子列作为特征
    """

    def __init__(self, models, risk_components,
                 hidden_dim: int = 64, lr: float = 3e-4,
                 n_epochs: int = 20, temperature: float = 1.0,
                 entropy_coef: float = 0.01,
                 feature_cols: Optional[List[str]] = None,
                 entry_threshold: float = 0.5, exit_threshold: float = -1.0,
                 name="RLPortfolio"):
        super().__init__(risk_components, entry_threshold, exit_threshold, name)
        self.models = models
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.temperature = temperature
        self.entropy_coef = entropy_coef
        self._feature_cols = feature_cols or []
        self.agent: Optional[RLAgent] = None
        self._trained = False
        self._train_metrics: List[dict] = []

    def train(self, dates, factor_df: pd.DataFrame,
              ret_series: pd.Series) -> List[dict]:
        feature_cols = self._feature_cols or list(factor_df.columns)
        self._feature_cols = feature_cols

        env = RLEnvironment(
            models=self.models,
            risk_components=self.risk_components,
            dates=dates,
            factor_df=factor_df,
            ret_series=ret_series,
            feature_cols=feature_cols,
        )

        self.agent = RLAgent(
            feature_dim=env.feature_dim,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            temperature=self.temperature,
            entropy_coef=self.entropy_coef,
        )

        all_metrics = []
        for epoch in range(self.n_epochs):
            state = env.reset()
            epoch_rewards = []
            for _ in range(len(dates) - 1):
                features = torch.from_numpy(state).float()
                weights_np, log_prob, value = self.agent.act_with_log_prob(features)
                next_state, reward, done, info = env.step(weights_np)
                self.agent.store(features, log_prob, value, reward)
                epoch_rewards.append(reward)
                if done:
                    break
                state = next_state

            metrics = self.agent.update()
            metrics["epoch"] = epoch + 1
            metrics["total_return"] = env.portfolio_value - 1.0
            metrics["portfolio_value"] = env.portfolio_value
            all_metrics.append(metrics)

        self._trained = True
        self._train_metrics = all_metrics
        return all_metrics

    @property
    def is_trained(self) -> bool:
        return self._trained

    def step(self, date, factor_df: pd.DataFrame,
             context: Dict[str, Any]) -> pd.Series:
        if self.agent is None:
            raise RuntimeError("RLPortfolio 未训练，请先调用 train()")

        if isinstance(factor_df.index, pd.MultiIndex):
            cross = factor_df.xs(date, level=0)
        else:
            cross = factor_df

        if len(cross) == 0:
            return pd.Series(dtype=float)

        # 构建特征（与训练时一致）
        features_list = []
        for m in self.models:
            try:
                pred = m.predict(cross)
                features_list.append(pred.values)
            except Exception:
                features_list.append(np.zeros(len(cross)))
        for col in self._feature_cols:
            if col in cross.columns:
                features_list.append(cross[col].values)
            else:
                features_list.append(np.zeros(len(cross)))

        features = np.column_stack(features_list).astype(np.float32)
        features = np.nan_to_num(features, 0.0)
        features_t = torch.from_numpy(features).float()

        weights_np = self.agent.act(features_t)
        w = pd.Series(weights_np, index=cross.index)

        # RL 权重为 softmax 概率，top_q 分位以上入池
        top_threshold = w.quantile(1 - max(self.entry_threshold / 10, 0.1))
        self.pool = set(w[w > top_threshold].index)

        # 只在池内分配
        pool_weights = w[w.index.isin(self.pool)]
        if len(pool_weights) == 0:
            return pd.Series(dtype=float)
        pool_weights = pool_weights / pool_weights.sum()

        # 风控链
        context["prev_weights"] = pd.Series(self.positions,
                                             index=pool_weights.index).fillna(0)
        pool_weights = self._apply_risk_chain(pool_weights, context)
        self.positions = {s: w for s, w in pool_weights.items() if abs(w) > 1e-8}
        self.pool = set(self.positions.keys())

        return pool_weights

    def save(self, path: str):
        if self.agent is not None:
            torch.save({
                "agent": {
                    "actor": self.agent.actor.state_dict(),
                    "critic": self.agent.critic.state_dict(),
                    "opt": self.agent.optimizer.state_dict(),
                },
                "feature_dim": self.agent.feature_dim,
                "hidden_dim": self.hidden_dim,
                "feature_cols": self._feature_cols,
            }, path)

    def load(self, path: str):
        ckpt = torch.load(path, weights_only=True)
        feat_dim = ckpt["feature_dim"]
        self._feature_cols = ckpt.get("feature_cols", [])
        self.agent = RLAgent(feature_dim=feat_dim,
                             hidden_dim=ckpt.get("hidden_dim", self.hidden_dim))
        self.agent.actor.load_state_dict(ckpt["agent"]["actor"])
        self.agent.critic.load_state_dict(ckpt["agent"]["critic"])
        self.agent.optimizer.load_state_dict(ckpt["agent"]["opt"])
        self._trained = True
