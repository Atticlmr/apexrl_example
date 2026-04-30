# ApexRL Genesis Examples

This repository contains ApexRL PPO examples migrated from Genesis demos:

- `go2_example`: Unitree Go2 locomotion.
- `drone_example`: Crazyflie drone hovering.

Both examples use ApexRL's `VecEnv` interface and custom `ContinuousActor` / `Critic` networks.

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

## Outputs

Each example writes logs and checkpoints under its own `logs/<experiment-name>/` directory. These outputs are ignored by git.
