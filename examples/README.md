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
| `closed_loop.py` | Closed-loop linkages from MJCF equality constraints |
| `authored_decomp.py` | A ring-stacking toy with an authored convex decomposition |
| `convex_decomposition.py`, `nonconvex_mesh.py` | Objects dropped into a non-convex tank |
| `contact_manifold.py` | The multi-contact manifold of a box on a box |
| `contype.py` | Contact filtering with contype and conaffinity bitmasks |
| `pyramid.py`, `tower.py` | Stacking stability |
| `friction_breakaway.py`, `rolling_coast.py`, `torsional_grasp.py` | Friction cones, rolling and torsional friction |
| `diffik_controller.py` | A differential inverse kinematics controller built on the Jacobian |
| `ik_shadow_hand.py` | Multi-link inverse kinematics on a Shadow Hand |
| `merge_entities.py` | An arm and its hand loaded from separate files and merged |
| `suction_cup.py` | A suction gripper made of a runtime weld constraint |
| `kinematic_go2_ghost.py` | A kinematic ghost showing a reference motion |
| `terrain_from_mesh.py`, `terrain_height_field.py`, `terrain_subterrain.py` | Height-field terrains |
| `drone_fly_route.py` | A PID-controlled drone flying through waypoints |
| `drone_teleop.py` | A drone flown open-loop from propeller speeds |
| `usd_fridge.py`, `usd_kitchen.py` | Articulated assets imported from USD stages (needs `usd-core`) |

## deformable

| Example | Shows |
|---|---|
| `fem_vertex_constraints.py` | Hard and soft vertex constraints on FEM entities, under the legacy or the SAP coupler |
| `pbd_flags_in_wind.py` | PBD cloth flags in wind, turbulence and a point attractor |
| `pbd_elastic_dragon.py` | A PBD elastic body |

## fluids

| Example | Shows |
|---|---|
| `mpm_flush_cubes.py` | MPM liquid jets from emitters flushing elastic objects |
| `pbd_liquid.py` | A PBD liquid block collapsing |

## multiphysics

| Example | Shows |
|---|---|
| `rigid_mpm_cut_dragon.py` | An elastic dragon sliced by a thin rigid cutter |
| `rigid_mpm_grasp_soft_cube.py` | A Franka grasping an MPM cube |
| `rigid_mpm_attachment.py` | MPM particles attached to a rigid link |
| `rigid_mpm_sand_wheel.py` | Sand pouring over wheels |
| `rigid_sph_cube.py`, `rigid_sph_water_wheel.py` | Rigid bodies in SPH water |
| `sph_mpm_duck.py` | An MPM duck floating in SPH water |
| `rigid_pbd_cloth_drape.py`, `rigid_pbd_cloth_attached.py` | PBD cloth draped on and attached to rigid bodies |
| `rigid_fem_cube_on_arm.py` | An FEM cube pinned to a Franka hand |
| `sap_franka_grasp_fem_sphere.py`, `sap_franka_grasp_rigid_cube.py`, `sap_fem_sphere_and_cube.py` | The SAP coupler |
| `ipc_objects_falling.py`, `ipc_franka_grasp_cube.py`, `ipc_cloth_teleop.py` | The IPC coupler (see `ipc_README.md`) |

## sensors

| Example | Shows |
|---|---|
| `imu_franka.py` | An IMU with noise, drift and delay on a Franka hand |
| `joint_torque_franka.py` | Joint torque sensing |
| `contact_force_go2.py` | Contact force sensors on the feet of a Go2 |
| `lidar_go2.py` | A lidar or a depth camera on a Go2 patrolling among obstacles |
| `depth_camera_deforming_mesh.py` | Depth cameras watching a mesh deformed every frame |
| `surface_distance_shadowhand.py` | Distance probes on the fingertips of a Shadow Hand |
| `tactile_sandbox.py`, `tactile_franka.py` | Tactile sensors |
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
| `grasp_env.py`, `grasp_train.py`, `grasp_eval.py`, `behavior_cloning.py` | Franka grasping, from reinforcement learning to vision-based behavior cloning |
| `hover_env.py`, `hover_train.py`, `hover_eval.py` | Drone hovering |

## interaction

| Example | Shows |
|---|---|
| `mouse_interaction.py` | Dragging objects with the mouse |
| `keyboard_teleop.py` | End-effector teleoperation from the keyboard |
| `mesh_point_selector.py` | A custom viewer plugin that selects points on meshes |
| `imgui_joint_control.py` | The ImGui overlay with joint sliders |
