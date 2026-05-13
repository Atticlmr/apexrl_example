"""Train Genesis Go2 walking with ApexRL SAC.

Example:
    cd go2_example
    ../../.venv/bin/python train_sac.py --backend gpu --num-envs 256 --gradient-steps 16 --total-timesteps 50000000
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import torch

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import genesis as gs
from apexrl.agent import OffPolicyRunner
from apexrl.algorithms.sac import SACConfig
from env.go2_env import Go2Env
from go2_config import get_go2_cfgs


def apply_sac_task_overrides(
    args: argparse.Namespace,
    env_cfg: dict,
    reward_cfg: dict,
    command_cfg: dict,
) -> None:
    """Tune the Go2 task defaults for off-policy SAC training."""
    env_cfg["termination_if_roll_greater_than"] = args.termination_roll
    env_cfg["termination_if_pitch_greater_than"] = args.termination_pitch
    env_cfg["termination_if_base_height_less_than"] = args.termination_base_height

    if args.command_mode == "forward":
        command_cfg["lin_vel_x_range"] = [0.3, 1.0]
        command_cfg["lin_vel_y_range"] = [0.0, 0.0]
        command_cfg["ang_vel_range"] = [0.0, 0.0]

    if args.sac_reward_profile == "locomotion":
        reward_cfg["tracking_sigma"] = 0.05
        reward_cfg["reward_scales"].update(
            {
                "tracking_lin_vel": 5.0,
                "tracking_ang_vel": 1.0,
                "lin_vel_z": -0.5,
                "base_height": -40.0,
                "action_rate": -0.001,
                "similar_to_default": -0.02,
            }
        )


def build_sac_cfg(args: argparse.Namespace) -> SACConfig:
    return SACConfig(
        gamma=args.gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        max_timesteps=args.total_timesteps,
        actor_learning_rate=args.actor_lr,
        critic_learning_rate=args.critic_lr,
        alpha_learning_rate=args.alpha_lr,
        optimizer=args.optimizer,
        max_grad_norm=args.max_grad_norm,
        auto_alpha=not args.fixed_alpha,
        init_alpha=args.init_alpha,
        target_entropy=args.target_entropy,
        actor_hidden_dims=args.actor_hidden_dims,
        critic_hidden_dims=args.critic_hidden_dims,
        activation=args.activation,
        layer_norm=args.layer_norm,
        use_tanh_squash=True,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_replay_buffer=args.save_replay_buffer,
        extra_log_keys=args.extra_log_keys,
        logger_backend=args.logger_backend,
        device="auto",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="go2-walking-sac-apexrl")
    parser.add_argument("-B", "--num-envs", type=int, default=256)
    parser.add_argument("--total-timesteps", type=int, default=50_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument(
        "--termination-roll",
        type=float,
        default=20.0,
        help="Roll angle termination threshold in degrees for SAC training.",
    )
    parser.add_argument(
        "--termination-pitch",
        type=float,
        default=20.0,
        help="Pitch angle termination threshold in degrees for SAC training.",
    )
    parser.add_argument(
        "--termination-base-height",
        type=float,
        default=0.2,
        help="Base height termination threshold in meters for SAC training.",
    )

    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-starts", type=int, default=50_000)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=16)
    parser.add_argument("--target-update-interval", type=int, default=1)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument(
        "--optimizer", choices=["adam", "adamw", "muon"], default="adam"
    )
    parser.add_argument("--max-grad-norm", type=float, default=10.0)

    parser.add_argument("--fixed-alpha", action="store_true")
    parser.add_argument("--init-alpha", type=float, default=0.02)
    parser.add_argument("--target-entropy", type=float, default=-6.0)
    parser.add_argument(
        "--command-mode",
        choices=["forward", "omni"],
        default="forward",
        help="Use an easier forward command distribution for SAC, or keep the PPO-style omni commands.",
    )
    parser.add_argument(
        "--sac-reward-profile",
        choices=["locomotion", "ppo"],
        default="locomotion",
        help="Use SAC-tuned reward scales or keep the PPO reward scales.",
    )

    parser.add_argument(
        "--actor-hidden-dims",
        type=int,
        nargs="+",
        default=[512, 256, 128],
    )
    parser.add_argument(
        "--critic-hidden-dims",
        type=int,
        nargs="+",
        default=[512, 256, 128],
    )
    parser.add_argument(
        "--activation",
        choices=["elu", "relu", "tanh", "leaky_relu"],
        default="elu",
    )
    parser.add_argument("--layer-norm", action="store_true")

    parser.add_argument("--log-interval", type=int, default=50_000)
    parser.add_argument("--save-interval", type=int, default=1_000_000)
    parser.add_argument(
        "--save-replay-buffer",
        action="store_true",
        help="Include the SAC replay buffer in checkpoints. This can make .pt files very large.",
    )
    parser.add_argument(
        "--extra-log-keys",
        nargs="*",
        default=["log", "time_outs", "terminated", "truncated", "reward_components"],
        help="Extras top-level keys to recursively log under extra/*. Use no values to disable.",
    )
    parser.add_argument("--logger-backend", type=str, default="tensorboard")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(
        backend=backend,
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=args.backend == "gpu",
    )

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_go2_cfgs()
    apply_sac_task_overrides(args, env_cfg, reward_cfg, command_cfg)
    sac_cfg = build_sac_cfg(args)
    log_dir = args.log_root / args.exp_name
    checkpoint_dir = log_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with (log_dir / "cfgs.pkl").open("wb") as f:
        pickle.dump(
            {
                "env_cfg": env_cfg,
                "obs_cfg": obs_cfg,
                "reward_cfg": reward_cfg,
                "command_cfg": command_cfg,
                "sac_cfg": sac_cfg,
            },
            f,
        )

    device = torch.device(
        "cuda" if args.backend == "gpu" and torch.cuda.is_available() else "cpu"
    )
    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.show_viewer,
    )
    runner = OffPolicyRunner(
        env=env,
        cfg=sac_cfg,
        algorithm="sac",
        log_dir=str(log_dir),
        save_dir=str(checkpoint_dir),
        device=device,
    )
    try:
        runner.learn(total_timesteps=args.total_timesteps)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
