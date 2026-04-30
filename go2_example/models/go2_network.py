"""Custom ApexRL actor and critic networks for Go2 locomotion."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from apexrl.models.base import ContinuousActor, Critic
from apexrl.utils import flatten_observation


def _orthogonal_init(module: nn.Module, gain: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def _build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation: str,
) -> nn.Sequential:
    activation_cls = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }.get(activation.lower(), nn.ELU)

    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation_cls())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class Go2Actor(ContinuousActor):
    """Continuous Gaussian policy for the 45-D Go2 observation."""

    def __init__(self, obs_space, action_space, cfg: dict[str, Any] | None = None):
        super().__init__(obs_space, action_space, cfg)
        cfg = cfg or {}
        obs_dim = int(torch.tensor(obs_space.shape).prod().item())
        hidden_dims = cfg.get("hidden_dims", [512, 256, 128])
        activation = cfg.get("activation", "elu")
        init_std = cfg.get("init_std", 1.0)

        self.policy = _build_mlp(obs_dim, hidden_dims, self.action_dim, activation)
        self.log_std = nn.Parameter(torch.ones(self.action_dim) * math.log(init_std))

        self.policy.apply(lambda module: _orthogonal_init(module, math.sqrt(2.0)))
        _orthogonal_init(self.policy[-1], 0.01)

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        return self.policy(flatten_observation(obs))

    def get_action_dist(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.distributions.Normal:
        mean = self.forward(obs)
        min_log_std = self.cfg.get("min_log_std", -5.0)
        max_log_std = self.cfg.get("max_log_std", 2.0)
        std = torch.exp(torch.clamp(self.log_std, min_log_std, max_log_std))
        return torch.distributions.Normal(mean, std)


class Go2Critic(Critic):
    """Value function network for Go2 locomotion."""

    def __init__(self, obs_space, cfg: dict[str, Any] | None = None):
        super().__init__(obs_space, cfg)
        cfg = cfg or {}
        obs_dim = int(torch.tensor(obs_space.shape).prod().item())
        hidden_dims = cfg.get("hidden_dims", [512, 256, 128])
        activation = cfg.get("activation", "elu")

        self.value = _build_mlp(obs_dim, hidden_dims, 1, activation)
        self.value.apply(lambda module: _orthogonal_init(module, math.sqrt(2.0)))
        _orthogonal_init(self.value[-1], 1.0)

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        return self.value(flatten_observation(obs)).squeeze(-1)

    def get_value(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(obs)

