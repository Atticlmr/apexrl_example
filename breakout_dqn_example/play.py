"""Play or record a trained ApexRL DQN policy on Breakout."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from atari_dqn import AtariDQN
from breakout_config import get_dqn_cfg
from env.atari_env import make_breakout_vec_env
from models.breakout_q_network import BreakoutQNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", default="breakout-dqn-apexrl")
    parser.add_argument("--env-id", default="ALE/Breakout-v5")
    parser.add_argument("--checkpoint", default="checkpoint_final.pt")
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-dir", type=Path, default=Path("videos"))
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument(
        "--full-episode",
        action="store_true",
        help="Do not reset on life loss. This can leave Breakout waiting for FIRE.",
    )
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
    record_dir = None
    if args.record:
        render_mode = "rgb_array"
        record_dir = args.record_dir / args.exp_name
        record_dir.mkdir(parents=True, exist_ok=True)

    env = make_breakout_vec_env(
        1,
        env_id=args.env_id,
        seed=args.seed,
        render_mode=render_mode,
        clip_rewards=False,
        terminal_on_life_loss=not args.full_episode,
        record_dir=record_dir,
        device="cpu",
    )
    cfg = get_dqn_cfg(buffer_size=1, learning_starts=1)
    cfg.dueling = args.dueling
    agent = AtariDQN(
        env=env,
        cfg=cfg,
        q_network_class=BreakoutQNetwork,
        obs_space=env.observation_space_gym,
        action_space=env.action_space_gym,
        q_network_cfg={"dueling": args.dueling, "replay_device": "cpu"},
        device=device,
    )
    checkpoint = resolve_checkpoint(args.checkpoint, args.log_root, args.exp_name)
    agent.load(str(checkpoint))
    print(f"Loaded checkpoint: {checkpoint}")

    obs = env.reset()
    episode_rewards = []
    current_reward = 0.0
    try:
        while len(episode_rewards) < args.episodes:
            action = agent.act(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            current_reward += float(reward[0].item())
            if bool(done[0].item()):
                episode_rewards.append(current_reward)
                print(f"Episode {len(episode_rewards)} reward: {current_reward:.1f}")
                current_reward = 0.0
    finally:
        env.close()

    if episode_rewards:
        mean_reward = sum(episode_rewards) / len(episode_rewards)
        print(f"Mean reward over {len(episode_rewards)} episodes: {mean_reward:.1f}")
    if record_dir is not None:
        print(f"Videos written under: {record_dir}")


if __name__ == "__main__":
    main()
