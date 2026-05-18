# MPE MAPPO Example

This example trains ApexRL MAPPO on a torch-vectorized Simple Spread MPE task.
The training environment keeps all parallel state in batched tensors instead of creating one PettingZoo environment per parallel instance.

Install dependencies from the repository root:

```bash
cd /RL_ws
source .venv/bin/activate
cd /RL_ws/apexrl_example
uv pip install -r requirements.txt
```

Train:

```bash
cd /RL_ws/apexrl_example/mpe_mappo_example
python train.py --device cuda -B 4096 --num-steps 16 --num-epochs 4 --minibatch-size 16384 --total-timesteps 100000000 --save-interval 20 -e mpe-simple-spread-mappo-4096
```

With 4096 parallel environments and 3 agents, each MAPPO iteration collects 196,608 agent-steps.

Quick CPU smoke test:

```bash
cd /RL_ws/apexrl_example/mpe_mappo_example
python train.py --device cpu -B 1 --total-timesteps 384 --num-steps 32 --minibatch-size 32 -e smoke-mpe
```

Play with rendering:

```bash
cd /RL_ws/apexrl_example/mpe_mappo_example
python play.py -e mpe-simple-spread-mappo-4096 --checkpoint checkpoint_final.pt --device cpu
```
