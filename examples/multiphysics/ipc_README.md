## IPC examples

The IPC (Incremental Potential Contact) coupler handles contact between cloth, deformable FEM entities and rigid links
in one solver, with no interpenetration.

**Prerequisites:**
1. Install the optional LibUIPC dependency: `pip install pyuipc`.
2. Use Linux or Windows on x86 with an Nvidia GPU. The IPC solver always runs on CUDA.

**Examples:**

1. **Folding a towel:**
   ```bash
   python examples/multiphysics/ipc_fold_towel.py -v
   ```
   Two Frankas pinch the corners of an IPC cloth towel and fold it in half by friction alone. With `-v`, the keyboard
   then teleoperates one Franka.

1. **Grasping soft entities:**
   ```bash
   python examples/multiphysics/soft_grasp.py --coupler ipc -v
   ```
   A Franka places a tomato, a jelly and a plush toy, all FEM entities.
