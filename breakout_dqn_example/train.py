"""Train ApexRL DQN on ALE/Breakout-v5."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
from apexrl.agent import OffPolicyRunner
from atari_dqn import AtariDQN
from breakout_config import get_dqn_cfg
from env.atari_env import make_breakout_vec_env
from models.breakout_q_network import BreakoutQNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", default="breakout-dqn-apexrl")
    parser.add_argument("--env-id", default="ALE/Breakout-v5")
    parser.add_argument("-B", "--num-envs", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=50_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument("--buffer-size", type=int, default=800_000)
    parser.add_argument("--learning-starts", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-interval", type=int, default=1_000_000)
    parser.add_argument("--dueling", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    cfg = get_dqn_cfg(
        total_timesteps=args.total_timesteps,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
    )
    cfg.dueling = args.dueling

    log_dir = args.log_root / args.exp_name
    checkpoint_dir = log_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = make_breakout_vec_env(
        args.num_envs,
        env_id=args.env_id,
        seed=args.seed,
        clip_rewards=True,
        terminal_on_life_loss=True,
        device="cpu",
    )
    with (log_dir / "cfgs.pkl").open("wb") as f:
        pickle.dump(
            {
                "env_id": args.env_id,
                "num_envs": args.num_envs,
                "seed": args.seed,
                "dqn_cfg": cfg,
            },
            f,
        )

    agent = AtariDQN(
        env=env,
        cfg=cfg,
        q_network_class=BreakoutQNetwork,
        obs_space=env.observation_space_gym,
        action_space=env.action_space_gym,
        q_network_cfg={"dueling": args.dueling, "replay_device": "cpu"},
        device=device,
    )
    runner = OffPolicyRunner(
        env=env,
        cfg=cfg,
        algorithm="dqn",
        agent=agent,
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
