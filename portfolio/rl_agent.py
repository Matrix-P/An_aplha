"""RL Agent: Actor-Critic (PyTorch)"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple


class PortfolioNet(nn.Module):
    """每只股票独立 MLP → 评分"""

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RLAgent:
    """Actor-Critic Agent"""

    def __init__(self, feature_dim: int, hidden_dim: int = 64,
                 lr: float = 3e-4, gamma: float = 0.99,
                 temperature: float = 1.0, entropy_coef: float = 0.01):
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.actor = PortfolioNet(feature_dim, hidden_dim)
        self.critic = PortfolioNet(feature_dim, hidden_dim)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self._reset_buffer()

    def _reset_buffer(self):
        self.states: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def act(self, features: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            raw_scores = self.actor(features) / self.temperature
            probs = torch.softmax(raw_scores, dim=0)
        return probs.cpu().numpy()

    def act_with_log_prob(self, features: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        raw_scores = self.actor(features) / self.temperature
        probs = torch.softmax(raw_scores, dim=0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        value = self.critic(features).mean()
        log_prob = dist.log_prob(action).sum()
        return probs.detach().cpu().numpy(), log_prob, value

    def store(self, features, log_prob, value, reward):
        self.states.append(features.detach())
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)

    def update(self) -> dict:
        if len(self.rewards) < 1:
            return {}

        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        values = torch.stack(self.values)
        advantages = returns - values
        log_probs = torch.stack(self.log_probs)
        policy_loss = -(log_probs * advantages.detach()).mean()

        probs_list = []
        for s in self.states:
            raw = self.actor(s) / self.temperature
            probs_list.append(torch.softmax(raw, dim=0))
        if probs_list:
            entropies = torch.stack([
                -(p * torch.log(p + 1e-12)).sum() for p in probs_list
            ])
            entropy_loss = -self.entropy_coef * entropies.mean()
        else:
            entropy_loss = torch.tensor(0.0)

        value_loss = advantages.pow(2).mean()
        loss = policy_loss + value_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.optimizer.step()
        self._reset_buffer()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item() / max(self.entropy_coef, 1e-8),
            "mean_return": returns.mean().item(),
        }
