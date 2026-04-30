"""Train Genesis drone hovering with ApexRL PPO."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

import genesis as gs
from apexrl.agent import OnPolicyRunner
from drone_config import get_drone_cfgs, get_ppo_cfg
from env.hover_env import HoverEnv
from models.drone_network import DroneActor, DroneCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="drone-hovering-apexrl")
    parser.add_argument("-B", "--num-envs", type=int, default=8192)
    parser.add_argument("--max-iterations", type=int, default=301)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument("--vis", action="store_true")
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

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_drone_cfgs()
    if args.vis:
        env_cfg["visualize_target"] = True
    ppo_cfg = get_ppo_cfg(num_envs=args.num_envs, max_iterations=args.max_iterations)
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
                "ppo_cfg": ppo_cfg,
            },
            f,
        )

    device = torch.device("cuda" if args.backend == "gpu" and torch.cuda.is_available() else "cpu")
    env = HoverEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )
    runner = OnPolicyRunner(
        env=env,
        cfg=ppo_cfg,
        algorithm="ppo",
        actor_class=DroneActor,
        critic_class=DroneCritic,
        log_dir=str(log_dir),
        save_dir=str(checkpoint_dir),
        device=device,
    )
    try:
        runner.learn(num_iterations=args.max_iterations)
    finally:
        runner.close()


if __name__ == "__main__":
    main()

