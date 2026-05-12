"""Play or record a trained Genesis Go2 ApexRL SAC policy."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="go2-walking-sac-apexrl")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_final.pt")
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="cpu")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lin-vel-x", type=float, default=None)
    parser.add_argument("--lin-vel-y", type=float, default=None)
    parser.add_argument("--ang-vel", type=float, default=None)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-path", type=Path, default=Path("video_sac.mp4"))
    parser.add_argument("--no-viewer", action="store_true")
    return parser.parse_args()


def resolve_checkpoint(log_dir: Path, checkpoint: str) -> Path:
    path = Path(checkpoint)
    if path.is_absolute():
        return path
    direct = log_dir / checkpoint
    if direct.exists():
        return direct.resolve()
    return (log_dir / "checkpoints" / checkpoint).resolve()


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

    log_dir = args.log_root / args.exp_name
    with (log_dir / "cfgs.pkl").open("rb") as f:
        cfgs = pickle.load(f)

    env_cfg = cfgs["env_cfg"]
    obs_cfg = cfgs["obs_cfg"]
    reward_cfg = cfgs["reward_cfg"]
    command_cfg = cfgs["command_cfg"]
    sac_cfg = cfgs.get("sac_cfg", SACConfig())

    reward_cfg = {**reward_cfg, "reward_scales": {}}
    env_cfg["visualize_camera"] = args.record
    env_cfg["max_visualize_FPS"] = 60

    if args.lin_vel_x is not None:
        command_cfg["lin_vel_x_range"] = [args.lin_vel_x, args.lin_vel_x]
    if args.lin_vel_y is not None:
        command_cfg["lin_vel_y_range"] = [args.lin_vel_y, args.lin_vel_y]
    if args.ang_vel is not None:
        command_cfg["ang_vel_range"] = [args.ang_vel, args.ang_vel]

    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.no_viewer,
    )
    device = torch.device("cuda" if args.backend == "gpu" and torch.cuda.is_available() else "cpu")
    runner = OffPolicyRunner(
        env=env,
        cfg=sac_cfg,
        algorithm="sac",
        log_dir=None,
        save_dir=str(log_dir / "checkpoints"),
        device=device,
    )
    checkpoint = resolve_checkpoint(log_dir, args.checkpoint)
    runner.load_checkpoint(str(checkpoint))
    runner.agent.actor.eval()
    runner.agent.critic1.eval()
    runner.agent.critic2.eval()

    obs = env.reset()
    max_steps = args.max_steps
    if args.record and max_steps <= 0:
        max_steps = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])

    try:
        with torch.no_grad():
            if args.record and env.cam is not None:
                env.cam.start_recording()
            step = 0
            while max_steps <= 0 or step < max_steps:
                actions = runner.agent.act(obs, deterministic=True)
                obs, _, _, _ = env.step(actions)
                if args.record and env.cam is not None:
                    env.cam.render()
                step += 1
            if args.record and env.cam is not None:
                env.cam.stop_recording(
                    save_to_filename=str(args.record_path),
                    fps=env_cfg["max_visualize_FPS"],
                )
    finally:
        runner.close()


if __name__ == "__main__":
    main()
