"""Batched Simple Spread environment for ApexRL MAPPO."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from apexrl.multiagent import MultiAgentVecEnv
from gymnasium import spaces


class MPEParallelVecEnv(MultiAgentVecEnv):
    """Torch-vectorized MPE Simple Spread compatible with ApexRL MAPPO."""

    def __init__(
        self,
        num_envs: int,
        *,
        num_agents: int = 3,
        max_cycles: int = 25,
        continuous_actions: bool = False,
        local_ratio: float = 0.5,
        coverage_reward_scale: float = 2.0,
        capture_radius: float = 0.1,
        seed: int = 1,
        render_mode: str | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if continuous_actions:
            raise NotImplementedError("This batched example supports discrete actions.")
        if not 0.0 <= local_ratio <= 1.0:
            raise ValueError("local_ratio must be in [0, 1]")

        self.num_envs = int(num_envs)
        self.num_agents = int(num_agents)
        self.num_landmarks = int(num_agents)
        self.max_cycles = int(max_cycles)
        self.local_ratio = float(local_ratio)
        self.coverage_reward_scale = float(coverage_reward_scale)
        self.capture_radius = float(capture_radius)
        self.seed = int(seed)
        self.render_mode = render_mode
        self.device = torch.device(device)
        self.possible_agents = [f"agent_{idx}" for idx in range(self.num_agents)]

        obs_dim = 4 + 2 * self.num_landmarks + 2 * (self.num_agents - 1)
        state_dim = obs_dim * self.num_agents
        self.observation_spaces = {
            agent_id: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )
            for agent_id in self.possible_agents
        }
        self.action_spaces = {
            agent_id: spaces.Discrete(5) for agent_id in self.possible_agents
        }
        self.state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        )

        self.dt = 0.1
        self.damping = 0.25
        self.accel = 5.0
        self.agent_size = 0.15
        self.contact_force = 100.0
        self.contact_margin = 1e-3

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)
        self.steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.agent_pos = torch.zeros(
            self.num_envs, self.num_agents, 2, device=self.device
        )
        self.agent_vel = torch.zeros_like(self.agent_pos)
        self.landmark_pos = torch.zeros(
            self.num_envs, self.num_landmarks, 2, device=self.device
        )
        self._episode_returns = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._episode_lengths = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._obs: dict[str, torch.Tensor] = {}
        self._state = torch.zeros(
            self.num_envs,
            state_dim,
            dtype=torch.float32,
            device=self.device,
        )

        self._pygame = None
        self._screen = None
        self._clock = None
        self._width = 700
        self._height = 700
        self.reset()

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        self._reset_envs(torch.arange(self.num_envs, device=self.device))
        return self.get_observations(), {}

    def _reset_envs(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        shape = (env_ids.numel(), self.num_agents, 2)
        landmark_shape = (env_ids.numel(), self.num_landmarks, 2)
        self.agent_pos[env_ids] = (
            torch.rand(shape, generator=self.generator, device=self.device) * 2.0 - 1.0
        )
        self.agent_vel[env_ids] = 0.0
        self.landmark_pos[env_ids] = (
            torch.rand(landmark_shape, generator=self.generator, device=self.device)
            * 2.0
            - 1.0
        )
        self.steps[env_ids] = 0
        self._episode_returns[env_ids] = 0.0
        self._episode_lengths[env_ids] = 0.0
        self._update_observations()

    def get_observations(self) -> dict[str, torch.Tensor]:
        return self._obs

    def get_state(self) -> torch.Tensor:
        return self._state

    def step(
        self,
        actions: dict[str, torch.Tensor],
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, Any],
    ]:
        action_tensor = torch.stack(
            [
                actions[agent_id]
                .to(device=self.device, dtype=torch.long)
                .reshape(self.num_envs)
                for agent_id in self.possible_agents
            ],
            dim=1,
        )
        self._integrate(action_tensor)
        rewards = self._compute_rewards()

        self.steps += 1
        done = self.steps >= self.max_cycles
        self._episode_returns += rewards.mean(dim=1)
        self._episode_lengths += 1.0

        completed_returns = self._episode_returns[done].detach().cpu().tolist()
        completed_lengths = self._episode_lengths[done].detach().cpu().tolist()

        terminated = {
            agent_id: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for agent_id in self.possible_agents
        }
        truncated = {agent_id: done.clone() for agent_id in self.possible_agents}

        if done.any():
            self._reset_envs(torch.where(done)[0])
        else:
            self._update_observations()

        reward_dict = {
            agent_id: rewards[:, agent_idx]
            for agent_idx, agent_id in enumerate(self.possible_agents)
        }
        extras = {
            "log": {
                "completed_return": completed_returns,
                "completed_length": completed_lengths,
                "mean_landmark_distance": self._mean_landmark_distance(),
                "collision_rate": self._collision_rate(),
            }
        }
        return self.get_observations(), reward_dict, terminated, truncated, extras

    def _integrate(self, action_tensor: torch.Tensor) -> None:
        force = torch.zeros_like(self.agent_pos)
        force[:, :, 0] += (action_tensor == 2).float()
        force[:, :, 0] -= (action_tensor == 1).float()
        force[:, :, 1] += (action_tensor == 4).float()
        force[:, :, 1] -= (action_tensor == 3).float()
        force *= self.accel
        force += self._collision_forces()

        self.agent_pos += self.agent_vel * self.dt
        self.agent_vel *= 1.0 - self.damping
        self.agent_vel += force * self.dt

    def _collision_forces(self) -> torch.Tensor:
        delta = self.agent_pos[:, :, None, :] - self.agent_pos[:, None, :, :]
        dist = torch.linalg.norm(delta, dim=-1).clamp_min(1e-6)
        dist_min = self.agent_size * 2.0
        penetration = (
            torch.nn.functional.softplus(-(dist - dist_min) / self.contact_margin)
            * self.contact_margin
        )
        force = (
            self.contact_force * delta / dist.unsqueeze(-1) * penetration.unsqueeze(-1)
        )
        eye = torch.eye(self.num_agents, dtype=torch.bool, device=self.device)
        force = force.masked_fill(eye.view(1, self.num_agents, self.num_agents, 1), 0.0)
        return force.sum(dim=2)

    def _compute_rewards(self) -> torch.Tensor:
        landmark_dists = torch.linalg.norm(
            self.agent_pos[:, :, None, :] - self.landmark_pos[:, None, :, :],
            dim=-1,
        )
        min_landmark_dists = landmark_dists.min(dim=1).values
        covered_landmarks = (
            (min_landmark_dists < self.capture_radius).float().sum(dim=1)
        )
        global_reward = (
            -min_landmark_dists.sum(dim=1)
            + self.coverage_reward_scale * covered_landmarks
        )

        agent_dists = torch.linalg.norm(
            self.agent_pos[:, :, None, :] - self.agent_pos[:, None, :, :],
            dim=-1,
        )
        eye = torch.eye(self.num_agents, dtype=torch.bool, device=self.device)
        collisions = (agent_dists < self.agent_size * 2.0) & ~eye.view(
            1, self.num_agents, self.num_agents
        )
        local_reward = -collisions.float().sum(dim=2)

        return (
            global_reward[:, None] * (1.0 - self.local_ratio)
            + local_reward * self.local_ratio
        )

    def _update_observations(self) -> None:
        obs_list = []
        for agent_idx, agent_id in enumerate(self.possible_agents):
            self_pos = self.agent_pos[:, agent_idx, :]
            self_vel = self.agent_vel[:, agent_idx, :]
            landmark_rel = (self.landmark_pos - self_pos[:, None, :]).reshape(
                self.num_envs,
                -1,
            )
            other_indices = [idx for idx in range(self.num_agents) if idx != agent_idx]
            other_rel = (
                self.agent_pos[:, other_indices, :] - self_pos[:, None, :]
            ).reshape(self.num_envs, -1)
            obs = torch.cat([self_vel, self_pos, landmark_rel, other_rel], dim=-1)
            self._obs[agent_id] = obs
            obs_list.append(obs)
        self._state = torch.cat(obs_list, dim=-1)

    def _mean_landmark_distance(self) -> torch.Tensor:
        landmark_dists = torch.linalg.norm(
            self.agent_pos[:, :, None, :] - self.landmark_pos[:, None, :, :],
            dim=-1,
        )
        return landmark_dists.min(dim=1).values.mean()

    def _collision_rate(self) -> torch.Tensor:
        agent_dists = torch.linalg.norm(
            self.agent_pos[:, :, None, :] - self.agent_pos[:, None, :, :],
            dim=-1,
        )
        eye = torch.eye(self.num_agents, dtype=torch.bool, device=self.device)
        collisions = (agent_dists < self.agent_size * 2.0) & ~eye.view(
            1, self.num_agents, self.num_agents
        )
        return collisions.float().mean()

    def render(self) -> Any:
        if self.render_mode is None:
            return None
        if self._pygame is None:
            import pygame

            pygame.init()
            self._pygame = pygame
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((self._width, self._height))
                self._clock = pygame.time.Clock()
            else:
                self._screen = pygame.Surface((self._width, self._height))

        pygame = self._pygame
        screen = self._screen
        assert screen is not None
        screen.fill((255, 255, 255))

        agent_pos = self.agent_pos[0].detach().cpu()
        landmark_pos = self.landmark_pos[0].detach().cpu()
        all_pos = torch.cat([agent_pos, landmark_pos], dim=0)
        cam_range = max(float(all_pos.abs().max().item()), 1.25)

        def to_screen(pos: torch.Tensor) -> tuple[int, int]:
            x = self._width / 2 + float(pos[0]) / cam_range * self._width * 0.42
            y = self._height / 2 - float(pos[1]) / cam_range * self._height * 0.42
            return int(x), int(y)

        landmark_radius = max(8, int(self.agent_size / cam_range * self._width * 0.3))
        agent_radius = max(10, int(self.agent_size / cam_range * self._width * 0.35))
        for landmark in landmark_pos:
            pygame.draw.circle(
                screen, (80, 80, 80), to_screen(landmark), landmark_radius
            )
            pygame.draw.circle(
                screen, (0, 0, 0), to_screen(landmark), landmark_radius, 1
            )
        for agent in agent_pos:
            pygame.draw.circle(screen, (70, 90, 220), to_screen(agent), agent_radius)
            pygame.draw.circle(screen, (0, 0, 0), to_screen(agent), agent_radius, 1)

        if self.render_mode == "rgb_array":
            frame = pygame.surfarray.array3d(screen)
            return np.transpose(frame, (1, 0, 2))
        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(10)
        return None

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
            self._screen = None
            self._clock = None
