# Reinforcement learning examples

Each task is an environment (`*_env.py`) with its training (`*_train.py`) and evaluation (`*_eval.py`) scripts. The
training scripts use the PPO implementation of `rsl-rl`.

## Go2 on terrain (`go2_terrain_env.py`, `go2_terrain_train.py`, `go2_terrain_eval.py`)

A Go2 learns to walk over rough ground, slopes, stairs and stepping stones. It observes a raycaster height scan and
foot contact sensors. Friction and trunk payload are randomized per environment, and a curriculum moves each
environment to harder terrain as it succeeds. Training takes about 10 minutes at 4096 environments on one GPU.

```bash
python examples/rl/go2_terrain_train.py
python examples/rl/go2_terrain_eval.py            # downloads the shipped policy
python examples/rl/go2_terrain_eval.py --ckpt 1499
```

The evaluation walks the robot the length of the mixed terrain row, with a kinematic ghost following the command.
With `-v`, the arrow keys set the speed and the heading.

## Drone hover (`hover_env.py`, `hover_train.py`, `hover_eval.py`)

The hover environment (`hover_env.py`) is designed to train a drone to maintain a stable hover position by reaching randomly generated target points. The environment includes:

 - Initialization of the scene and entities (plane, drone and target).
 - Reward functions to provide feedback to the agent based on its performance in reaching the target points.
 - **Command resampling to generate new random target points** and environment reset functionalities to ensure continuous training.

**Acknowledgement**: The reward design is inspired by [Champion-level drone racing using deep
reinforcement learning (Nature 2023)](https://www.nature.com/articles/s41586-023-06419-4.pdf)

### Installation

At this stage, we have defined the environments. Now, we use the PPO implementation from `rsl-rl` to train the policy. First, install all Python dependencies via `pip`:

```bash
pip install tensorboard "rsl-rl-lib>=5.0.0"
```

### Training

Train the drone hovering policy using the `HoverEnv` environment.

Run with:

```bash
python examples/rl/hover_train.py -e drone-hovering -b 8192 --max-iterations 301
```

Train with visualization:

```bash
python examples/rl/hover_train.py -e drone-hovering -b 8192 --max-iterations 301 -v
```

### Evaluation

Evaluate the trained drone hovering policy.

Run with:

```bash
python examples/rl/hover_eval.py -e drone-hovering --ckpt 300 --record
```

**Note**: If you experience slow performance or encounter other issues
during evaluation, try removing the `--record` option.

For the latest updates, detailed documentation, and additional resources, visit this repository: [GenesisDroneEnv](https://github.com/KafuuChikai/GenesisDroneEnv).
