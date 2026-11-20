"""
Walk a trained Go2 across rough ground, a slope, stairs and stepping stones, beside a ghost of its commanded motion.

The robot starts on the mixed row of the terrain and heads along it, through the four tiles. A translucent kinematic Go2
follows the commanded velocity and heading from the pose of the robot every two seconds, with the joint targets of the
policy, so that the gap between the two shows how well the policy tracks its command. Without --ckpt, the policy shipped
with Genesis is downloaded. With the viewer (-v), the walk runs, then the keyboard steers the robot:

    Up / Down       speed up / slow down
    Left / Right    turn left / right
"""

import argparse
import math
import os
import pickle

import torch
from huggingface_hub import snapshot_download

import genesis as gs
from genesis.utils.geom import xyz_to_quat
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind

from go2_terrain_env import Go2TerrainEnv
from go2_terrain_train import get_cfgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="go2-terrain", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument(
        "--ckpt", type=int, default=None, help="Checkpoint iteration to load instead of the shipped one"
    )
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record a video of the walk to 'out/go2_terrain_eval.mp4'"
    )
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32")

    if args.ckpt is not None:
        log_dir = f"logs/{args.exp_name}"
        with open(f"{log_dir}/cfgs.pkl", "rb") as f:
            env_cfg, obs_cfg, reward_cfg, command_cfg, _ = pickle.load(f)
        checkpoint = torch.load(f"{log_dir}/model_{args.ckpt}.pt", map_location=gs.device, weights_only=True)
        policy_state = {
            name.removeprefix("mlp."): value
            for name, value in checkpoint["actor_state_dict"].items()
            if name.startswith("mlp.")
        }
    else:
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
        asset_dir = snapshot_download(
            repo_type="dataset",
            repo_id="Genesis-Intelligence/assets",
            revision="5bab5bf83036173395880074b5237ce498a4a763",
            allow_patterns="rl/go2_terrain/*",
        )
        policy_state = torch.load(
            os.path.join(asset_dir, "rl", "go2_terrain", "walk.pt"), map_location=gs.device, weights_only=True
        )

    # The walk is one long episode with a fixed command, steered by its heading target alone.
    speed = 0.8
    env_cfg["episode_length_s"] = 1e6
    env_cfg["resampling_time_s"] = 1e6
    env_cfg["max_init_level"] = 0
    command_cfg["lin_vel_x_range"] = [speed, speed]
    command_cfg["lin_vel_y_range"] = [0.0, 0.0]
    reward_cfg["reward_scales"] = {}

    env = Go2TerrainEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
        ghost=True,
        record=args.record,
    )

    policy = torch.nn.Sequential(
        torch.nn.Linear(env.obs_buf.shape[-1], 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, env.num_actions),
    )
    policy.load_state_dict(policy_state)
    policy.to(device=gs.device)

    # Start at the near end of the mixed row, which is the last level, heading along it through the four tiles.
    env.reset()
    start_xy = torch.tensor([[(env.n_levels - 0.5) * env.tile_size, 1.5]], dtype=gs.tc_float, device=gs.device)
    start_z = env.terrain.get_terrain_height(start_xy)[..., 0] + env_cfg["base_init_height"]
    start_pos = torch.cat((start_xy, start_z[:, None]), dim=-1)
    start_quat = xyz_to_quat(torch.tensor([[0.0, 0.0, math.pi / 2]], dtype=gs.tc_float, device=gs.device))
    env.robot.set_qpos(torch.cat((start_pos, start_quat, env.default_dof_pos[None]), dim=-1), zero_velocity=True)
    env.spawn_pos.copy_(start_pos)
    env.heading_target.fill_(math.pi / 2)

    def set_speed(delta):
        env.commands[:, 0] = torch.clip(env.commands[:, 0] + delta, 0.0, 1.0)

    def turn(delta):
        env.heading_target += delta

    if args.vis:
        env.scene.viewer.register_keybinds(
            Keybind("speed_up", Key.UP, KeyAction.PRESS, callback=set_speed, args=(0.1,)),
            Keybind("slow_down", Key.DOWN, KeyAction.PRESS, callback=set_speed, args=(-0.1,)),
            Keybind("turn_left", Key.LEFT, KeyAction.HOLD, callback=turn, args=(0.05,)),
            Keybind("turn_right", Key.RIGHT, KeyAction.HOLD, callback=turn, args=(-0.05,)),
        )
        gs.logger.info("Keyboard: [Up/Down] speed up / slow down, [Left/Right] turn left / right")

    # The ghost restarts from the pose of the robot every two seconds, then integrates the command.
    ghost_period = round(2.0 / env.dt)
    ghost_height = 0.32
    ghost_pos = start_pos.clone()
    ghost_euler = torch.zeros((1, 3), dtype=gs.tc_float, device=gs.device)

    def step(i_step, obs):
        if i_step % ghost_period == 0:
            ghost_pos.copy_(env.base_pos)
            ghost_euler[:, 2] = env.heading
        yaw = ghost_euler[:, 2]
        ghost_pos[:, 0] += (env.commands[:, 0] * torch.cos(yaw) - env.commands[:, 1] * torch.sin(yaw)) * env.dt
        ghost_pos[:, 1] += (env.commands[:, 0] * torch.sin(yaw) + env.commands[:, 1] * torch.cos(yaw)) * env.dt
        ghost_euler[:, 2] += env.commands[:, 2] * env.dt
        ghost_pos[:, 2] = env.terrain.get_terrain_height(ghost_pos[:, None, :2])[..., 0] + ghost_height
        env.ghost.set_qpos(torch.cat((ghost_pos, xyz_to_quat(ghost_euler), env.dof_pos_target), dim=-1))
        with torch.no_grad():
            obs, _, _, _ = env.step(policy(obs["policy"]))
        return obs

    camera_lookat = tensor_to_array(start_pos[0])
    camera_pos = camera_lookat + (-2.5, -2.5, 1.6)
    if args.vis:
        env.scene.viewer.set_camera_pose(pos=camera_pos, lookat=camera_lookat)
        env.scene.viewer.follow_entity(env.robot, smoothing=0.9)
    if env.camera is not None:
        env.camera.set_pose(pos=camera_pos, lookat=camera_lookat)
        env.camera.follow_entity(env.robot, smoothing=0.9)
        env.camera.start_recording(save_to_filename="out/go2_terrain_eval.mp4", fps=round(1.0 / env.dt))
    obs = env.get_observations()
    # Four tiles at the walking speed, from one end of the row to the other.
    n_steps = round((env.n_tiles * env.tile_size - 3.0) / speed / env.dt) if "PYTEST_VERSION" not in os.environ else 5
    for i_step in range(n_steps):
        obs = step(i_step, obs)
    if env.camera is not None:
        env.camera.stop_recording()
    i_step = n_steps
    while args.vis and env.scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        obs = step(i_step, obs)
        i_step += 1


if __name__ == "__main__":
    main()
