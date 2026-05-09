"""Atari Breakout environment construction for ApexRL DQN."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from apexrl.envs.gym_wrapper import GymVecEnv
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo


class ClipRewardEnv(gym.RewardWrapper):
    """Clip Atari rewards to {-1, 0, 1}, as in the original DQN setup."""

    def reward(self, reward: float) -> float:
        return float(np.sign(reward))


class FireResetEnv(gym.Wrapper):
    """Start Atari games that require FIRE before gameplay begins."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        if len(meanings) < 3 or meanings[1] != "FIRE":
            raise ValueError(f"FireResetEnv requires action 1 to be FIRE, got {meanings}")

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


def _register_ale() -> None:
    try:
        import ale_py
    except ImportError as exc:
        raise ImportError(
            "Atari environments require ale-py. Install it with "
            "`pip install \"gymnasium[atari,accept-rom-license]\" ale-py`."
        ) from exc
    gym.register_envs(ale_py)


def make_breakout_env(
    env_id: str = "ALE/Breakout-v5",
    *,
    seed: int | None = None,
    render_mode: str | None = None,
    clip_rewards: bool = True,
    terminal_on_life_loss: bool = True,
    record_dir: str | Path | None = None,
) -> gym.Env:
    """Create one preprocessed Breakout environment."""
    _register_ale()
    env = gym.make(
        env_id,
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
        render_mode=render_mode,
    )
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=terminal_on_life_loss,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    env = FireResetEnv(env)
    env = FrameStackObservation(env, stack_size=4)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if record_dir is not None:
        env = RecordVideo(
            env,
            video_folder=str(record_dir),
            episode_trigger=lambda episode_id: True,
            name_prefix="breakout",
        )
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def make_breakout_vec_env(
    num_envs: int,
    *,
    env_id: str = "ALE/Breakout-v5",
    seed: int = 1,
    render_mode: str | None = None,
    clip_rewards: bool = True,
    terminal_on_life_loss: bool = True,
    record_dir: str | Path | None = None,
    device: str = "cpu",
) -> GymVecEnv:
    """Create an ApexRL GymVecEnv for Breakout."""
    env_fns: list[Callable[[], gym.Env]] = []
    for index in range(num_envs):
        env_seed = seed + index
        env_record_dir = record_dir if index == 0 else None
        env_fns.append(
            lambda env_seed=env_seed, env_record_dir=env_record_dir: make_breakout_env(
                env_id=env_id,
                seed=env_seed,
                render_mode=render_mode,
                clip_rewards=clip_rewards,
                terminal_on_life_loss=terminal_on_life_loss,
                record_dir=env_record_dir,
            )
        )
    return GymVecEnv(env_fns, device=device)
