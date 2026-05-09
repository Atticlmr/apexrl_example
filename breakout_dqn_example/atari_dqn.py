"""Atari-specific DQN tweaks for image replay buffers."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from apexrl.algorithms.dqn import DQN, DQNConfig
from apexrl.buffer.replay_buffer import ReplayBuffer
from apexrl.optimizers import build_optimizer
from apexrl.utils import observation_to_device, space_to_spec


class AtariDQN(DQN):
    """DQN with uint8 CPU replay storage for Atari image observations."""

    def __init__(
        self,
        *args: Any,
        cfg: DQNConfig | None = None,
        q_network_cfg: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        cfg = cfg or DQNConfig()
        q_network_cfg = q_network_cfg or {}
        self.replay_device = torch.device(q_network_cfg.get("replay_device", "cpu"))

        real_buffer_size = cfg.buffer_size
        cfg_for_super = copy.copy(cfg)
        cfg_for_super.buffer_size = 1
        super().__init__(*args, cfg=cfg_for_super, q_network_cfg=q_network_cfg, **kwargs)
        self.cfg.buffer_size = real_buffer_size
        self.replay_buffer = ReplayBuffer(
            capacity=real_buffer_size,
            obs_shape=space_to_spec(self.obs_space),
            action_shape=(),
            device=self.replay_device,
            obs_dtype=torch.uint8,
            action_dtype=torch.long,
        )
        self.optimizer = build_optimizer(
            self.cfg.optimizer,
            lr=self.cfg.learning_rate,
            modules=self.q_network,
        )

    def update(self) -> dict[str, float]:
        if len(self.replay_buffer) < max(self.cfg.batch_size, self.cfg.learning_starts):
            return {}

        batch = self.replay_buffer.sample(self.cfg.batch_size)
        observations = observation_to_device(batch["observations"], self.device)
        next_observations = observation_to_device(batch["next_observations"], self.device)
        actions = batch["actions"].to(self.device, dtype=torch.long)
        rewards = batch["rewards"].to(self.device, dtype=torch.float32)
        dones = batch["dones"].to(self.device, dtype=torch.float32)

        q_values = self.q_network(observations)
        chosen_q = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            if self.cfg.double_dqn:
                next_actions = self.q_network(next_observations).argmax(dim=-1, keepdim=True)
                next_q = self.target_q_network(next_observations).gather(1, next_actions).squeeze(-1)
            else:
                next_q = self.target_q_network(next_observations).max(dim=-1).values
            td_target = rewards + self.cfg.gamma * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(chosen_q, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            self.cfg.max_grad_norm,
        )
        self.optimizer.step()

        self.num_updates += 1
        self._maybe_update_target_network()

        return {
            "train/q_loss": loss.item(),
            "train/mean_q": chosen_q.mean().item(),
            "train/td_target_mean": td_target.mean().item(),
            "train/grad_norm": float(grad_norm),
            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def save(self, path: str) -> None:
        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_q_network_state_dict": self.target_q_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "total_timesteps": self.total_timesteps,
            "num_updates": self.num_updates,
            "config": self.cfg,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.iteration = checkpoint.get("iteration", 0)
        self.total_timesteps = checkpoint.get("total_timesteps", 0)
        self.num_updates = checkpoint.get("num_updates", 0)
