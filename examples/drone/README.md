# Drone Examples

This directory contains examples of drone simulations using the Genesis framework.

## Available Examples

### 1. Interactive Drone (`interactive_drone.py`)
An open-loop drone flight driven by differential rotor RPMs. Headless, it replays a short scripted sequence of key
commands: climb, tilt forward and back, then turn left and back. With `-v`, the same sequence runs first, then you fly
the drone with the keyboard:
- Up: Tilt Forward (North)
- Down: Tilt Backward (South)
- Left: Turn Left
- Right: Turn Right
- Space: Increase Thrust (Accelerate)
- Left Shift: Decrease Thrust (Decelerate)
- ESC: Quit

Run with:
```bash
python interactive_drone.py -v
```

Add `-r out.mp4` to save a video of the run.

### 2. Waypoint Flight (`fly_route.py`)
A cascaded PID controller, written in the script, provides stable point-to-point flight without controlling each
rotor. Its gains are tuned for controlled flight but not optimized. `fly_route.py` flies the drone through a
route of three waypoints. With `-v`, the arrow keys move the target in the ground plane and K/J move it up and down,
during the route and after it.

Run with:
```bash
python fly_route.py -v
```

Add `-r out.mp4` to save a video of the run.

### 3. Hover Environment (`hover_env.py`, `hover_train.py`, `hover_eval.py`)

The hover environment (`hover_env.py`) is designed to train a drone to maintain a stable hover position by reaching randomly generated target points. The environment includes:

 - Initialization of the scene and entities (plane, drone and target).
 - Reward functions to provide feedback to the agent based on its performance in reaching the target points.
 - **Command resampling to generate new random target points** and environment reset functionalities to ensure continuous training.

**Acknowledgement**: The reward design is inspired by [Champion-level drone racing using deep
reinforcement learning (Nature 2023)](https://www.nature.com/articles/s41586-023-06419-4.pdf)

#### 3.0 Installation

At this stage, we have defined the environments. Now, we use the PPO implementation from `rsl-rl` to train the policy. First, install all Python dependencies via `pip`:

```bash
pip install tensorboard "rsl-rl-lib>=5.0.0"
```

#### 3.1 Training

Train the drone hovering policy using the `HoverEnv` environment.

Run with:

```bash
python hover_train.py -e drone-hovering -b 8192 --max-iterations 301
```

Train with visualization:

```bash
python hover_train.py -e drone-hovering -b 8192 --max-iterations 301 -v
```

#### 3.2 Evaluation

Evaluate the trained drone hovering policy.

Run with:

```bash
python hover_eval.py -e drone-hovering --ckpt 300 --record
```

**Note**: If you experience slow performance or encounter other issues
during evaluation, try removing the `--record` option.

For the latest updates, detailed documentation, and additional resources, visit this repository: [GenesisDroneEnv](https://github.com/KafuuChikai/GenesisDroneEnv).

## Technical Details

- The drone model used is the Crazyflie 2.X (`urdf/drones/cf2x.urdf`)
- Base hover RPM is approximately 14468
- Movement is achieved by varying individual rotor RPMs to create directional thrust
- The simulation uses realistic physics including gravity and aerodynamics

## Controls Implementation

The interactive drone uses differential RPM control:
- Forward/Backward tilt: Adjusts front/back rotor pairs
- Left/Right turn: Speeds up one diagonal rotor pair and slows the other
- The control is open-loop: a tilt held then released leaves the drone rotating until the opposite key stops it
- RPM changes are automatically clipped to safe ranges (0-25000 RPM)