"""Configuration helpers for Breakout DQN."""

from __future__ import annotations

from apexrl.algorithms.dqn import DQNConfig


def get_dqn_cfg(
    *,
    total_timesteps: int | None = None,
    buffer_size: int = 100_000,
    learning_starts: int = 50_000,
    batch_size: int = 32,
    save_interval: int = 500_000,
    train_freq: int = 1,
    gradient_steps: int = 2,
) -> DQNConfig:
    return DQNConfig(
        learning_rate=1e-4,
        gamma=0.99,
        batch_size=batch_size,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=1_000,
        max_grad_norm=10.0,
        double_dqn=True,
        dueling=False,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=1_000_000,
        max_timesteps=total_timesteps,
        log_interval=10_000,
        save_interval=save_interval,
        logger_backend="tensorboard",
    )
