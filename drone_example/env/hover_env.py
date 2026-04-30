"""Genesis drone hovering environment using ApexRL's VecEnv interface."""

from __future__ import annotations

import copy
import math
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import genesis as gs
from apexrl.envs.vecenv import TensorDict, VecEnv
from genesis.utils.geom import (
    inv_quat,
    quat_to_xyz,
    transform_by_quat,
    transform_quat_by_quat,
)


def gs_rand_float(lower: float, upper: float, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class HoverEnv(VecEnv):
    """Vectorized Genesis Crazyflie hover task compatible with ApexRL PPO."""

    def __init__(
        self,
        num_envs: int,
        env_cfg: dict[str, Any],
        obs_cfg: dict[str, Any],
        reward_cfg: dict[str, Any],
        command_cfg: dict[str, Any],
        show_viewer: bool = False,
    ):
        super().__init__(device=gs.device)
        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)
        self.num_actions = env_cfg["num_actions"]
        self.num_obs = 17
        self.num_privileged_obs = 0
        self.cfg = env_cfg
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device
        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

        self.observation_space_gym = gym.spaces.Dict(
            {
                "obs": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_obs,),
                    dtype=np.float32,
                )
            }
        )
        self.action_space_gym = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.rendered_env_num))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())

        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.5, 0.5))
                ),
            )
        else:
            self.target = None

        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )
        else:
            self.cam = None

        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))
        self.scene.build(n_envs=num_envs)

        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_bool)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)
        self.base_euler = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.rel_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.last_rel_pos = torch.zeros_like(self.rel_pos)
        self.crash_condition = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_bool)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.extras: dict[str, Any] = {}

        self.reset()

    def get_observations(self) -> TensorDict:
        return TensorDict({"obs": self.obs_buf}, batch_size=[self.num_envs])

    def get_privileged_observations(self) -> None:
        return None

    def reset(self) -> TensorDict:
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        self._update_observation()
        return self.get_observations()

    def reset_idx(self, env_ids: torch.Tensor) -> TensorDict | None:
        if env_ids.numel() == 0:
            return None
        env_ids = env_ids.to(device=gs.device, dtype=torch.long)
        self.base_pos[env_ids] = self.base_init_pos
        self.last_base_pos[env_ids] = self.base_init_pos
        self.base_quat[env_ids] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(self.base_pos[env_ids], zero_velocity=True, envs_idx=env_ids)
        self.drone.set_quat(self.base_quat[env_ids], zero_velocity=True, envs_idx=env_ids)
        self.base_lin_vel[env_ids] = 0.0
        self.base_ang_vel[env_ids] = 0.0
        self.drone.zero_all_dofs_velocity(env_ids)
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = True

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][env_ids] = 0.0

        self._resample_commands(env_ids)
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        return None

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        self.actions = torch.clip(
            actions.to(device=gs.device, dtype=gs.tc_float),
            -self.env_cfg["clip_actions"],
            self.env_cfg["clip_actions"],
        )
        exec_actions = self.actions
        self.drone.set_propellers_rpm((1.0 + exec_actions * 0.8) * 14468.429183500699)
        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True)
        self.scene.step()

        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        reached_target = self._at_target()

        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
            | (torch.abs(self.rel_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
        )
        terminated = self.crash_condition
        time_outs = (self.episode_length_buf > self.max_episode_length) & ~terminated
        dones = time_outs | terminated

        self.rew_buf[:] = 0.0
        reward_components: dict[str, torch.Tensor] = {}
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            reward_components[name] = rew.detach()

        self._update_observation()
        final_observation = TensorDict({"obs": self.obs_buf.clone()}, batch_size=[self.num_envs])
        self.extras = {
            "time_outs": time_outs,
            "terminated": terminated,
            "truncated": time_outs,
            "final_observation": final_observation,
            "reward_components": reward_components,
            "log": {
                "/drone/target_distance": torch.norm(self.rel_pos, dim=1).mean(),
                "/drone/height": self.base_pos[:, 2].mean(),
                "/drone/crash_rate": terminated.float().mean(),
                "/drone/target_reached_rate": reached_target.numel() / self.num_envs,
            },
        }

        if dones.any():
            self.reset_idx(dones.nonzero(as_tuple=False).reshape((-1,)))
        active_reached_target = reached_target[~dones[reached_target]]
        if active_reached_target.numel() > 0:
            self._resample_commands(active_reached_target)
            self.rel_pos[active_reached_target] = self.commands[active_reached_target] - self.base_pos[active_reached_target]
            self.last_rel_pos[active_reached_target] = (
                self.commands[active_reached_target] - self.last_base_pos[active_reached_target]
            )
        if dones.any() or active_reached_target.numel() > 0:
            self._update_observation()

        self.last_actions[:] = self.actions[:]
        self.reset_buf = dones
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def close(self) -> None:
        if hasattr(self.scene, "viewer") and self.scene.viewer is not None:
            self.scene.viewer.stop()

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.commands[env_ids, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(env_ids),), gs.device)
        self.commands[env_ids, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(env_ids),), gs.device)
        self.commands[env_ids, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(env_ids),), gs.device)

    def _at_target(self) -> torch.Tensor:
        return (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"])
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )

    def _update_observation(self) -> None:
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1.0, 1.0),
                self.base_quat,
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1.0, 1.0),
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1.0, 1.0),
                self.last_actions,
            ],
            dim=-1,
        )

    def _reward_target(self) -> torch.Tensor:
        last_dist = torch.norm(self.last_rel_pos, dim=1)
        dist = torch.norm(self.rel_pos, dim=1)
        progress = last_dist - dist
        target_sigma = self.reward_cfg.get("target_sigma", 0.35)
        progress_weight = self.reward_cfg.get("target_progress_weight", 4.0)
        bonus = self.reward_cfg.get("target_bonus", 2.0)
        proximity = torch.exp(-dist / target_sigma)
        target_bonus = (dist < self.env_cfg["at_target_threshold"]).float() * bonus
        return progress_weight * progress + proximity + target_bonus

    def _reward_smooth(self) -> torch.Tensor:
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_yaw(self) -> torch.Tensor:
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180.0, yaw - 360.0, yaw) / 180.0 * 3.14159
        return torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))

    def _reward_angular(self) -> torch.Tensor:
        return torch.norm(self.base_ang_vel / 3.14159, dim=1)

    def _reward_crash(self) -> torch.Tensor:
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1.0
        return crash_rew
