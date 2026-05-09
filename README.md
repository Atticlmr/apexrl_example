# ApexRL Genesis Examples

This repository contains ApexRL PPO examples migrated from Genesis demos:

- `go2_example`: Unitree Go2 locomotion.
- `drone_example`: Crazyflie drone hovering.
- `breakout_dqn_example`: Atari Breakout DQN.

The Genesis examples use ApexRL's `VecEnv` interface and custom `ContinuousActor` / `Critic` networks.
The Breakout example uses ApexRL's DQN with a custom CNN Q-network.

## Install

Create or activate a Python environment first. From the parent workspace used here:

```bash
cd /RL_ws
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
pip install -r requirements.txt
```

If your setup still cannot find the Atari ROMs, install them with:

```bash
pip install autorom
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
python train.py --device cuda -B 8 --total-timesteps 10000000 --buffer-size 500000 -e breakout-dqn-apexrl
```

Quick CPU smoke test:

```bash
cd /RL_ws/apexrl_example/breakout_dqn_example
python train.py --device cpu -B 1 --total-timesteps 100 --learning-starts 1000 --buffer-size 1000 --save-interval 1000 -e smoke-breakout
```

Play:

```bash
cd /RL_ws/apexrl_example/breakout_dqn_example
python play.py -e breakout-dqn-apexrl --checkpoint checkpoint_final.pt --device cpu
```

Record a video:

```bash
cd /RL_ws/apexrl_example/breakout_dqn_example
python play.py -e breakout-dqn-apexrl --checkpoint checkpoint_final.pt --device cpu --record --no-render
```

Breakout is a pixel-based Atari task, so it usually needs far more than a few thousand steps before the policy looks good. Start with the smoke test only to verify the pipeline.

## Outputs

Each example writes logs and checkpoints under its own `logs/<experiment-name>/` directory. These outputs are ignored by git.
