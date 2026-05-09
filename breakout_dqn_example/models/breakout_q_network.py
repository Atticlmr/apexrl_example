"""CNN Q-network for Atari Breakout."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from apexrl.models.base import DiscreteQNetwork
from gymnasium import spaces


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class BreakoutQNetwork(DiscreteQNetwork):
    """Nature-DQN style convolutional Q-network for stacked Atari frames."""

    def __init__(
        self,
        obs_space: spaces.Box,
        action_space: spaces.Discrete,
        cfg: dict[str, Any] | None = None,
    ):
        super().__init__(obs_space, action_space, cfg)
        cfg = cfg or {}
        if len(obs_space.shape) != 3:
            raise ValueError(f"Expected 3D image observation, got {obs_space.shape}")

        self.channels_first = obs_space.shape[0] in (1, 4)
        in_channels = obs_space.shape[0] if self.channels_first else obs_space.shape[-1]
        self.dueling = bool(cfg.get("dueling", False))

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, *obs_space.shape)
            if not self.channels_first:
                sample = sample.permute(0, 3, 1, 2)
            feature_dim = int(self.encoder(sample).shape[-1])

        self.feature = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
        )
        if self.dueling:
            self.value_head = nn.Linear(512, 1)
            self.advantage_head = nn.Linear(512, self.num_actions)
        else:
            self.q_head = nn.Linear(512, self.num_actions)

        self.apply(lambda module: _orthogonal_init(module, gain=2**0.5))
        if self.dueling:
            _orthogonal_init(self.value_head)
            _orthogonal_init(self.advantage_head)
        else:
            _orthogonal_init(self.q_head)

    def _format_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        if not self.channels_first:
            obs = obs.permute(0, 3, 1, 2)
        return obs.float() / 255.0

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(obs, dict):
            obs = obs["obs"]
        features = self.feature(self.encoder(self._format_obs(obs)))
        if not self.dueling:
            return self.q_head(features)
        values = self.value_head(features)
        advantages = self.advantage_head(features)
        return values + advantages - advantages.mean(dim=-1, keepdim=True)
