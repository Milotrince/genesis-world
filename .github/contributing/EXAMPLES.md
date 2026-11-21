# Examples Reference

Location: `examples/`, one folder per feature area. Every script lives in a folder; nothing sits at the root.

## Running Examples

```bash
uv run examples/tutorials/01_hello_genesis.py
uv run examples/rigid/wrecking_ball.py -v
```

## Command-line convention

Every example parses its arguments the same way, so learning one script's CLI teaches all of them.

**Short flags are a closed, reserved set.** Each letter means exactly one thing across the whole tree. A flag
that is not in this table gets no short form.

| Short | Long | Type | Default |
|-------|------|------|---------|
| `-v` | `--vis` | `store_true` | off |
| `-g` | `--gpu` | `store_true` | off (CPU) |
| `-b` | `--num-envs` | `int` | per script |
| `-s` | `--steps` | `int` | per script |
| `-t` | `--seconds` | `float` | per script |
| `-e` | `--exp-name` | `str` | per script |
| `-r` | `--record` | `store_true` | off |
| `-o` | `--output-dir` | `str` | per script |
| `-d` | `--debug` | `store_true` | off |

Rules:

1. Short flag first, then long: `parser.add_argument("-v", "--vis", ...)`.
2. Long names use dashes between words (`--num-envs`); argparse exposes them as the underscored
   `args.num_envs`, so no `dest=` is needed.
3. Single-dash flags are one character.
4. Booleans are `store_true` and default off, which makes `default=False` redundant. Where the negative form
   reads better, a single `store_true` negative flag covers it (`--no-ipc`, `--no-force`).
5. Use `--steps` when the script's natural unit is iterations, `-t/--seconds` when it reasons in simulated
   time (that is, it also exposes `--dt` or compares against `scene.get_time()`).
6. `--dt` is the only spelling for the timestep; it matches `SimOptions(dt=...)`.
7. Every flag carries a `help=`. Keyword order is `action=` / `type=`, then `default=`, then `help=`.
8. The parser is named `parser`, built as the first statements of `main()`, with
   `if __name__ == "__main__": main()` at the bottom.

## Backend

Every example runs on CPU unless it has a stated reason not to, so a bare `python examples/...` works on any
machine. A GPU only pays off past roughly fifty parallel environments, which almost no example builds, and
`gs.init()` left to itself prefers a GPU, so the backend is always spelled out rather than left implicit.

A script that genuinely needs a GPU - a batched training or benchmark run, or a renderer that only exists on
CUDA - hardcodes `backend=gs.gpu` and says why in a comment, rather than exposing a flag with a single usable
value. Everything else takes `-g/--gpu` as an opt-in, or hardcodes `backend=gs.cpu` when it has no parser.
The same rule covers a script that only works on CPU.

Long-running scripts cut their horizon under `"PYTEST_VERSION" in os.environ` so the examples CI stays cheap.

## Behaviour

Every example outside `tutorials/` behaves the same way in three modes, so a user can run any of them without reading
it first.

- **Headless by default.** Without `-v`, the script runs a scripted scenario that shows its feature end to end, then
  exits. It never idles or waits for input.
- **`-r/--record`** adds a camera framed like the viewer and records the run to `out/<script>.mp4`: the scripted
  scenario headless, the interactive session with `-v`.
- **Interactive with `-v`.** The scripted scenario is skipped, and the user controls the scene from the first frame
  while the viewer is alive. Scenes about objects attach `MouseInteractionPlugin`, so the user can drag them. Scenes
  about an agent (a robot, a drone) register keyboard controls, avoid the keys of the default viewer controls, and log
  their keymap at start. The module docstring lists the same keymap.

Branch on `args.vis` once the scene is built: the interactive `while scene.viewer.is_alive()` loop under `-v`, the
scripted `for` loops otherwise. `camera.stop_recording()` follows whichever ran. Write a scripted phase as the same
calls the keys make, one phase after the other.

The examples about the viewer itself (`viewer_plugin/`, `gui/`) always open the viewer, since it is their subject.
Tutorials stay minimal and linear, and take `-v` only to open the viewer.

## Output files

Everything an example writes goes under `out/`, resolved against the directory the script is launched from, so
a single ignore rule covers the whole tree and nothing lands next to the sources. Name the file after the
example rather than something generic, since every script shares the one directory. Where a script exposes
`-o/--output-dir`, default it to a subdirectory of `out/`. Training scripts are the exception: they keep
checkpoints under `logs/<exp-name>/`, which their evaluation counterparts read back.

## Style

Examples are read as reference code, so they are held to `CODING_GUIDELINES.md` like the rest of the tree.
Beyond it, two things matter most here:

- **Keep a script linear.** A helper function is warranted when it is called more than once, or is passed as
  a callback or thread target. A one-shot linear script reads top to bottom inside `main()`.
- **Model the sanctioned data access.** Reach for the public entity and solver getters first, and
  `qd_to_numpy` / `qd_to_torch` from `genesis.utils.misc` when no getter exists.

## Key Examples

### Getting Started
- `tutorials/01_hello_genesis.py` - Basic introduction
- `tutorials/03_control_your_robot.py` - Robot control basics
- `tutorials/05_parallel_simulation.py` - Batched environments

### Robotics
- `rigid/diffik_controller.py` - Differential inverse kinematics
- `rigid/ik_shadow_hand.py` - Multi-link inverse kinematics
- `parallel/domain_randomization.py` - Domain randomization

### Multi-Physics
- `deformable/pbd_tablecloth.py` - Cloth-rigid coupling
- `fluids/sph_pour.py` - Fluid-rigid coupling
- `multiphysics/rigid_mpm_sand_wheel.py` - Granular-rigid coupling

### Training
- `rl/go2_env.py` - Quadruped RL environment
- `rl/grasp_env.py` - Grasping RL environment
