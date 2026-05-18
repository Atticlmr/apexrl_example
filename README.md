# ApexRL Genesis Examples

This repository contains ApexRL PPO examples migrated from Genesis demos:

- `go2_example`: Unitree Go2 locomotion.
- `drone_example`: Crazyflie drone hovering.
- `breakout_dqn_example`: Atari Breakout DQN.
- `mpe_mappo_example`: Batched MPE Simple Spread MAPPO.

The Genesis examples use ApexRL's `VecEnv` interface and custom `ContinuousActor` / `Critic` networks.
The Breakout example uses ApexRL's DQN with a custom CNN Q-network.
The MPE example uses ApexRL's multi-agent `MultiAgentVecEnv` and a torch-vectorized Simple Spread environment, so large parallel batches such as 4096 envs do not create thousands of Python PettingZoo env objects.

## Install

Create or activate a Python environment first. From the parent workspace used here:

```bash
cd /RL_ws
uv venv .venv
source .venv/bin/activate
```

Install ApexRL and Genesis into that environment. If you are working from local source checkouts:

```bash
pip install -e /RL_ws/Apex_rl
pip install -e /RL_ws/Genesis
```

If your copies live elsewhere, replace those paths with your local ApexRL and Genesis paths.

Verify the install:

```bash
python -c "import apexrl, genesis; print(apexrl.__file__); print(genesis.__file__)"
```

For the Atari Breakout example, install Gymnasium's Atari dependencies:

```bash
cd /RL_ws/apexrl_example
uv pip install -r requirements.txt
```

If your setup still cannot find the Atari ROMs, install them with:

```bash
uv pip install autorom
AutoROM --accept-license
```

## Go2 Locomotion

Demo video: [go2.mp4](./go2_example/go2.mp4)

Train:

```bash
cd /RL_ws/apexrl_example/go2_example
python train.py --backend gpu -B 4096 --max-iterations 1000 -e go2-walking-apexrl
```


Play with Genesis viewer enabled:

```bash
cd /RL_ws/apexrl_example/go2_example
python play.py -e go2-walking-apexrl --checkpoint checkpoint_final.pt --backend cpu
```

Record a video:

```bash
python play.py -e go2-walking-apexrl --checkpoint checkpoint_final.pt --backend cpu --record
```

Train with SAC:

```bash
cd /RL_ws/apexrl_example/go2_example
python train_sac.py --backend gpu -e go2-walking-sac
```

SAC checkpoints do not save the replay buffer by default. Add `--save-replay-buffer` only if you explicitly need replay state in the `.pt` files.

Play SAC with Genesis viewer enabled:

```bash
cd /RL_ws/apexrl_example/go2_example
python play_sac.py -e go2-walking-sac --checkpoint checkpoint_final.pt --backend cpu
```

## Drone Hovering

Demo video: [crayflie.mp4](./drone_example/crayflie.mp4)

Train:

```bash
cd /RL_ws/apexrl_example/drone_example
python train.py --backend gpu -B 8192 --max-iterations 301 -e drone-hovering-apexrl
```

Quick CPU smoke test:

```bash
cd /RL_ws/apexrl_example/drone_example
python train.py --backend cpu -B 1 --max-iterations 1 -e smoke-drone
```

Play with Genesis viewer enabled:

```bash
cd /RL_ws/apexrl_example/drone_example
python play.py -e drone-hovering-apexrl --checkpoint checkpoint_final.pt --backend cpu
```

Record a video:

```bash
cd /RL_ws/apexrl_example/drone_example
python play.py -e drone-hovering-apexrl --checkpoint checkpoint_final.pt --backend cpu --record
```

## Atari Breakout DQN

Train:

```bash
cd /RL_ws/apexrl_example/breakout_dqn_example
python train.py --device cuda -B 8 --total-timesteps 50000000 --buffer-size 500000 -e breakout-dqn-apexrl-v2
```

Quick CPU smoke test:

```bash
cd /RL_ws/apexrl_example/breakout_dqn_example
python train.py --device cpu -B 1 --total-timesteps 100 --learning-starts 1000 --buffer-size 1000 --save-interval 1000 -e smoke-breakout
```

Play:

```bash
cd /RL_ws/apexrl_example/breakout_dqn_example
python play.py -e breakout-dqn-apexrl-v2 --checkpoint checkpoint_final.pt --device cpu
```

Record a video:

```bash
cd /RL_ws/apexrl_example/breakout_dqn_example
python play.py -e breakout-dqn-apexrl-v2 --checkpoint checkpoint_final.pt --device cpu --record --no-render
```

Breakout is a pixel-based Atari task, so it usually needs far more than a few thousand steps before the policy looks good. Start with the smoke test only to verify the pipeline.

## MPE Simple Spread MAPPO

Train:

```bash
cd /RL_ws/apexrl_example/mpe_mappo_example
python train.py --device cuda -B 4096 --num-steps 16 --num-epochs 4 --minibatch-size 16384 --total-timesteps 100000000 --save-interval 20 -e mpe-simple-spread-mappo-4096
```

With 4096 parallel environments and 3 agents, this collects 196,608 agent-steps per MAPPO iteration.

Quick CPU smoke test:

```bash
cd /RL_ws/apexrl_example/mpe_mappo_example
python train.py --device cpu -B 1 --total-timesteps 384 --num-steps 32 --minibatch-size 32 -e smoke-mpe
```

Play:

```bash
cd /RL_ws/apexrl_example/mpe_mappo_example
python play.py -e mpe-simple-spread-mappo-4096 --checkpoint checkpoint_final.pt --device cpu
```

## Outputs

Each example writes logs and checkpoints under its own `logs/<experiment-name>/` directory. These outputs are ignored by git.
