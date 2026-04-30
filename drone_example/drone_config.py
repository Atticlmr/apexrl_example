"""Configuration for Genesis drone hovering with ApexRL PPO."""

from __future__ import annotations

from copy import deepcopy

from apexrl.algorithms.ppo import PPOConfig


def get_drone_cfgs() -> tuple[dict, dict, dict, dict]:
    env_cfg = {
        "num_actions": 4,
        "termination_if_roll_greater_than": 180.0,
        "termination_if_pitch_greater_than": 180.0,
        "termination_if_close_to_ground": 0.1,
        "termination_if_x_greater_than": 3.0,
        "termination_if_y_greater_than": 3.0,
        "termination_if_z_greater_than": 2.0,
        "base_init_pos": [0.0, 0.0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 15.0,
        "at_target_threshold": 0.1,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
    }
    obs_cfg = {
        "obs_scales": {
            "rel_pos": 1.0 / 3.0,
            "lin_vel": 1.0 / 3.0,
            "ang_vel": 1.0 / 3.14159,
        },
    }
    reward_cfg = {
        "yaw_lambda": -10.0,
        "target_sigma": 0.35,
        "target_progress_weight": 10.0,
        "target_bonus": 2.0,
        "reward_scales": {
            "target": 10.0,
            "smooth": -1e-4,
            "yaw": 0.01,
            "angular": -2e-4,
            "crash": -10.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "pos_x_range": [-1.0, 1.0],
        "pos_y_range": [-1.0, 1.0],
        "pos_z_range": [1.0, 1.0],
    }
    return deepcopy(env_cfg), deepcopy(obs_cfg), deepcopy(reward_cfg), deepcopy(command_cfg)


def get_ppo_cfg(num_envs: int, max_iterations: int | None = None) -> PPOConfig:
    num_steps = 100
    batch_size = num_steps * num_envs
    return PPOConfig(
        num_steps=num_steps,
        num_epochs=5,
        batch_size=batch_size,
        minibatch_size=max(batch_size // 4, 1),
        max_iterations=max_iterations,
        learning_rate=3e-4,
        max_learning_rate=3e-4,
        min_learning_rate=1e-5,
        learning_rate_schedule="adaptive",
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=1.0,
        ent_coef=0.004,
        max_grad_norm=1.0,
        target_kl=0.01,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="tanh",
        fixed_std=False,
        std_value=1.0,
        use_asymmetric=False,
        normalize_advantages=True,
        save_interval=100,
        log_interval=10,
        logger_backend="tensorboard",
    )
