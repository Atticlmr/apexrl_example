"""Play a trained ApexRL MAPPO policy on PettingZoo MPE simple_spread_v3."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from apexrl.models import MLPCritic, MLPDiscreteActor
from apexrl.multiagent import MAPPO, MAPPOConfig
from apexrl.multiagent.utils import multiagent_to_tensor
from env.mpe_vec_env import MPEParallelVecEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", default="mpe-simple-spread-mappo")
    parser.add_argument("--checkpoint", default="checkpoint_final.pt")
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--local-ratio", type=float, default=0.0)
    parser.add_argument("--coverage-reward-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(device_arg)


def resolve_checkpoint(path: str, log_root: Path, exp_name: str) -> Path:
    checkpoint = Path(path)
    if checkpoint.exists():
        return checkpoint
    candidate = log_root / exp_name / "checkpoints" / checkpoint
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint} or {candidate}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    render_mode = None if args.no_render else "human"

    cfg = MAPPOConfig(
        num_steps=1,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[256, 256],
        activation="tanh",
        centralized_critic=True,
        share_actor=True,
        share_critic=True,
        shared_reward=True,
        device=str(device),
    )
    env = MPEParallelVecEnv(
        1,
        num_agents=args.num_agents,
        max_cycles=args.max_cycles,
        local_ratio=args.local_ratio,
        coverage_reward_scale=args.coverage_reward_scale,
        seed=args.seed,
        render_mode=render_mode,
        device="cpu",
    )
    agent = MAPPO(
        env=env,
        cfg=cfg,
        actor_class=MLPDiscreteActor,
        critic_class=MLPCritic,
        device=device,
    )
    checkpoint = resolve_checkpoint(args.checkpoint, args.log_root, args.exp_name)
    agent.load(str(checkpoint))
    agent.eval()
    print(f"Loaded checkpoint: {checkpoint}")

    episode_rewards = []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            for _ in range(args.max_cycles):
                obs_t = multiagent_to_tensor(obs, device)
                with torch.no_grad():
                    actions, _ = agent.act(obs_t, deterministic=True)
                obs, rewards, terminated, truncated, _ = env.step(actions)
                total_reward += float(
                    torch.stack([rewards[agent_id] for agent_id in env.possible_agents])
                    .mean()
                    .item()
                )
                if not args.no_render:
                    env.render()
                done = torch.stack(
                    [
                        terminated[agent_id] | truncated[agent_id]
                        for agent_id in env.possible_agents
                    ]
                ).any()
                if bool(done.item()):
                    break
            episode_rewards.append(total_reward)
            print(f"Episode {episode + 1} reward: {total_reward:.2f}")
    finally:
        env.close()

    if episode_rewards:
        mean_reward = sum(episode_rewards) / len(episode_rewards)
        print(f"Mean reward over {len(episode_rewards)} episodes: {mean_reward:.2f}")


if __name__ == "__main__":
    main()
