# Examples

Each folder holds the examples of one engine capability. A file name starts with the solver, coupler or sensor the
example highlights, followed by the scene. Run any example from the repository root:

```bash
python examples/rigid/wrecking_ball.py        # headless: runs the scenario, then exits
python examples/rigid/wrecking_ball.py -v     # opens the viewer, which stays interactive
python examples/rigid/wrecking_ball.py -r     # records the run to out/wrecking_ball.mp4
```

Outside `tutorials/`, every example runs a scripted scenario headless, records it with `-r`, and stays interactive in
the viewer with `-v`, through the mouse or the keyboard. The examples in `interaction/` always open the viewer. See
`.github/contributing/EXAMPLES.md` for the conventions.

## tutorials

Read these in order. Each one teaches one API.

| Example | Shows |
|---|---|
| `01_hello_genesis.py` | A Franka falling onto a plane, the minimal scene |
| `02_visualization.py` | A domino run filmed by a moving camera, with video recording and RGB, depth and segmentation images |
| `03_control_your_robot.py` | Position, velocity and force control against setting the state, a Franka pushing a domino gently, then hard enough to start the run |
| `04_pick_and_place.py` | Inverse kinematics, motion planning and grasping, a Franka standing dominoes up and knocking them over |
| `05_parallel_simulation.py` | 512 domino runs with one spacing each, batched reads and the selection of rendered environments |
| `06_batched_ik.py` | Inverse kinematics across parallel environments |
| `07_draw_debug.py` | Debug drawing primitives |
| `08_entity_name.py` | Naming entities and looking them up |
| `09_ik_multilink.py` | Inverse kinematics on several links at once |
| `10_mpm.py` | MPM elastic, plastic and liquid materials |
| `11_pbd_cloth.py` | PBD cloth with pinned particles |
| `12_sph_liquid.py` | SPH liquid |
| `13_muscle.py` | MPM and FEM muscles |
| `14_worm.py` | A soft worm crawling with muscle groups |
| `15_hybrid_robot.py` | A robot with a rigid skeleton and a soft skin |
| `16_interactive_debugging.py` | Inspecting a scene from an IPython shell |

## rigid

| Example | Shows |
|---|---|
| `wrecking_ball.py` | A wrecking ball on a contact-held chain smashing a pile of cubes |
| `rube_goldberg.py` | A chain reaction through revolute, prismatic and spherical joints, with hibernation |
| `bolt_nut_self_screw.py` | A nut screwing itself down a bolt through non-convex thread contact |
| `screw_constraint.py` | A nut driven along a bolt by a screw constraint |
| `delta_robot.py` | A delta robot, a closed kinematic chain, sorting boxes off a conveyor |
| `stack_rings.py` | A Franka stacking the rings of a toy with an authored convex decomposition |
| `bin_picking.py` | A Franka picking household objects out of a non-convex bin, with fingers or suction |
| `contype.py` | Contact filtering with contype and conaffinity bitmasks |
| `jenga.py` | A Franka poking blocks out of a tower, with or without the contact patch |
| `mini_golf.py` | A Franka putting on greens of different rolling friction |
| `friction_breakaway.py`, `torsional_grasp.py` | Friction cones and torsional friction |
| `diffik_controller.py` | A differential inverse kinematics controller built on the Jacobian |
| `ik_shadow_hand.py` | Multi-link inverse kinematics on a Shadow Hand |
| `merge_entities.py` | An arm and its hand loaded from separate files and merged |
| `terrain_from_mesh.py`, `terrain_height_field.py` | Height-field terrains from a mesh and from an array |
| `drone_fly_route.py` | A PID-controlled drone flying through waypoints |
| `drone_teleop.py` | A drone flown open-loop from propeller speeds |
| `usd_fridge.py`, `usd_kitchen.py` | Articulated assets imported from USD stages (needs `usd-core`) |

## deformable

| Example | Shows |
|---|---|
| `fem_vertex_constraints.py` | Hard and soft vertex constraints on FEM entities, under the legacy or the SAP coupler |
| `mpm_roll_dough.py` | A Franka rolling elasto-plastic dough out into a sheet |
| `mpm_ice_cream.py` | Soft-serve emitted from a dispenser and coiled onto a cone the Franka holds |
| `pbd_flags_in_wind.py` | PBD cloth flags in wind, turbulence and a point attractor |
| `pbd_tablecloth.py` | Two Frankas draping a PBD tablecloth over a table |
| `pbd_jelly_delivery.py` | A Franka carrying a tray of PBD elastic jellies that wobble when it stops |

## fluids

| Example | Shows |
|---|---|
| `sph_pour.py` | A Franka pouring a cup of SPH water into a bowl |
| `mpm_hose.py` | A Franka hosing debris off a table with an MPM liquid jet |
| `pbd_liquid.py` | A PBD liquid block collapsing |

## multiphysics

| Example | Shows |
|---|---|
| `rigid_mpm_fruit_ninja.py` | A Franka slicing falling MPM fruit with a thin blade |
| `rigid_mpm_snowplow.py` | A wheeled robot plowing MPM snow into a berm |
| `rigid_mpm_sand_wheel.py` | Sand pouring over wheels |
| `sph_mpm_bathtub.py` | A rubber duck in a bathtub that fills from a faucet and drains through a vortex |
| `rigid_fem_cube_on_arm.py` | An FEM cube pinned to a Franka hand |
| `soft_grasp.py` | A Franka placing a tomato, a jelly and a plush toy under the SAP or the IPC coupler |
| `ipc_fold_towel.py` | Two Frankas folding a towel under the IPC coupler (see `ipc_README.md`) |

## sensors

| Example | Shows |
|---|---|
| `imu_franka.py` | An IMU with noise, drift and delay on a Franka hand |
| `joint_torque_franka.py` | Joint torque sensing |
| `contact_force_go2.py` | Contact force sensors on the feet of a Go2 |
| `lidar_go2.py` | A lidar or a depth camera on a Go2 patrolling among obstacles |
| `depth_camera_deforming_mesh.py` | Depth cameras watching a mesh deformed every frame |
| `surface_distance_shadowhand.py` | Distance probes on the fingertips of a Shadow Hand |
| `tactile_sandbox.py` | Five tactile sensor kinds on a pad pressed by four objects |
| `tactile_peg_insertion.py` | A Franka finding a hole by a force-guided spiral search and inserting a peg, with fingertip gels |
| `temperature_grid.py` | Heat transfer through contact |
| `camera_backends.py` | Camera sensors on every rendering backend |

## rendering

| Example | Shows |
|---|---|
| `raytracer_materials.py` | Surface materials under the raytracer (needs LuisaRender) |
| `batch_renderer_franka.py` | The batch renderer across parallel environments (needs CUDA and gs_madrona) |
| `rasterizer_render_async.py` | Simulating in a background thread while the viewer renders |
| `rasterizer_follow_entity.py`, `rasterizer_moving_camera.py` | Moving cameras |
| `rasterizer_speed_test.py` | Rasterizer throughput |

## differentiable

| Example | Shows |
|---|---|
| `rigid_sysid_launch.py` | The launch of a sliding box recovered from its trajectory through rigid gradients |
| `rigid_throw_to_target.py` | A Franka throw into a bin, tuned by gradient descent through the swing |
| `mpm_push.py` | Gradients through an MPM push, optimized with torch |

## parallel

| Example | Shows |
|---|---|
| `franka_heterogeneous_grasp.py` | A different object in each environment |
| `franka_grasp_bottle.py` | Batched grasping with motion planning |
| `domain_randomization.py`, `set_phys_attr.py` | Physical properties set per environment |
| `multi_gpu.py`, `ddp_multi_gpu.py` | Several GPUs |
| `benchmark_anymal_c.py`, `benchmark_franka.py` | Throughput at tens of thousands of environments (needs a GPU) |

## rl

Environments with their training and evaluation scripts, which need `rsl-rl`. See `rl/README.md`.

| Example | Shows |
|---|---|
| `go2_env.py`, `go2_train.py`, `go2_eval.py` | Go2 locomotion |
| `go2_backflip.py` | Pretrained Go2 backflips, runnable without training |
| `go2_terrain_env.py`, `go2_terrain_train.py`, `go2_terrain_eval.py` | A Go2 walking stairs, slopes and stepping stones from a height scan, with a terrain curriculum, randomized friction and payload, and a kinematic ghost of the command |
| `grasp_env.py`, `grasp_train.py`, `grasp_eval.py`, `behavior_cloning.py` | Franka grasping, from reinforcement learning to vision-based behavior cloning |
| `hover_env.py`, `hover_train.py`, `hover_eval.py` | Drone hovering |

## interaction

| Example | Shows |
|---|---|
| `mouse_interaction.py` | Dragging objects with the mouse |
| `keyboard_teleop.py` | End-effector teleoperation from the keyboard |
| `mesh_point_selector.py` | A custom viewer plugin that selects points on meshes |
| `imgui_joint_control.py` | The ImGui overlay with joint sliders |
