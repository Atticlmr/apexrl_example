"""Configuration for Genesis Go2 walking with ApexRL PPO."""

from __future__ import annotations

from copy import deepcopy

from apexrl.algorithms.ppo import PPOConfig


def get_go2_cfgs() -> tuple[dict, dict, dict, dict]:
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 10.0,
        "termination_if_pitch_greater_than": 10.0,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-0.5, 1.0],
        "lin_vel_y_range": [0.3, 0.3],
        "ang_vel_range": [-1.0, 1.0],
    }
    return deepcopy(env_cfg), deepcopy(obs_cfg), deepcopy(reward_cfg), deepcopy(command_cfg)


def get_ppo_cfg(num_envs: int, max_iterations: int | None = None) -> PPOConfig:
    num_steps = 24
    batch_size = num_steps * num_envs
    return PPOConfig(
        num_steps=num_steps,
        num_epochs=5,
        batch_size=batch_size,
        minibatch_size=max(batch_size // 4, 1),
        max_iterations=max_iterations,
        learning_rate=1e-3,
        max_learning_rate=1e-3,
        min_learning_rate=1e-5,
        learning_rate_schedule="adaptive",
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=1.0,
        ent_coef=0.01,
        max_grad_norm=1.0,
        target_kl=0.01,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        fixed_std=False,
        std_value=1.0,
        use_asymmetric=True,
        normalize_advantages=True,
        save_interval=100,
        log_interval=10,
        logger_backend="tensorboard",
    )
