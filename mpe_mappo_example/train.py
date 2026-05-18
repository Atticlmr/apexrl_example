"""Train ApexRL MAPPO on PettingZoo MPE simple_spread_v3."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
from apexrl.models import MLPCritic, MLPDiscreteActor
from apexrl.multiagent import MAPPO, MAPPOConfig, MultiAgentRunner
from env.mpe_vec_env import MPEParallelVecEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", default="mpe-simple-spread-mappo")
    parser.add_argument("-B", "--num-envs", type=int, default=16)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--local-ratio", type=float, default=0.0)
    parser.add_argument("--coverage-reward-scale", type=float, default=2.0)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=1)
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

    cfg = MAPPOConfig(
        num_steps=args.num_steps,
        num_epochs=args.num_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        learning_rate_schedule="constant",
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[256, 256],
        activation="tanh",
        centralized_critic=True,
        share_actor=True,
        share_critic=True,
        shared_reward=True,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        extra_log_keys=["log"],
        log_episode_metrics_vs_iteration=True,
        device=str(device),
    )

    log_dir = args.log_root / args.exp_name
    checkpoint_dir = log_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = MPEParallelVecEnv(
        args.num_envs,
        num_agents=args.num_agents,
        max_cycles=args.max_cycles,
        local_ratio=args.local_ratio,
        coverage_reward_scale=args.coverage_reward_scale,
        seed=args.seed,
        device="cpu",
    )
    with (log_dir / "cfgs.pkl").open("wb") as f:
        pickle.dump(
            {
                "num_envs": args.num_envs,
                "num_agents": args.num_agents,
                "max_cycles": args.max_cycles,
                "local_ratio": args.local_ratio,
                "coverage_reward_scale": args.coverage_reward_scale,
                "seed": args.seed,
                "mappo_cfg": cfg,
            },
            f,
        )

    agent = MAPPO(
        env=env,
        cfg=cfg,
        actor_class=MLPDiscreteActor,
        critic_class=MLPCritic,
        device=device,
    )
    runner = MultiAgentRunner(
        env=env,
        agent=agent,
        cfg=cfg,
        log_dir=str(log_dir),
        save_dir=str(checkpoint_dir),
        device=device,
    )
    try:
        runner.learn(total_timesteps=args.total_timesteps)
    finally:
        env.close()


if __name__ == "__main__":
    main()
