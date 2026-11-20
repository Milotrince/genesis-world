"""
Go2 locomotion over subterrains, observed through a height scan and foot contacts, under a terrain curriculum.

The terrain is one grid of subterrains: each row is a difficulty level (flat, rough, slope, stairs, stepping stones,
then all four mixed), and every tile is a pit or a pyramid that the robot climbs out of from its center. An environment
that walks out of its tile moves one level up, and one that barely moves moves one level down. Friction and the payload
of the trunk are randomized per environment.
"""

import math

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, xyz_to_quat


def gs_rand(lower, upper, batch_shape):
    return (upper - lower) * torch.rand(size=batch_shape, dtype=gs.tc_float, device=gs.device) + lower


class Go2TerrainEnv:
    def __init__(
        self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, ghost=False, record=False
    ):
        self.num_envs = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.device = gs.device

        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        # rsl-rl reads the environment configuration back as `cfg`.
        self.cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                max_collision_pairs=30,
                # Per-environment link masses need batched link info.
                batch_links_info=True,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-2.5, -2.5, 2.0),
                camera_lookat=(0.0, 0.0, 0.3),
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )

        # The collider holds a single height field, so every difficulty level is a row of one terrain.
        self.tile_size = env_cfg["tile_size"]
        self.terrain_types = env_cfg["terrain_types"]
        self.n_levels, self.n_tiles = len(self.terrain_types), len(self.terrain_types[0])
        self.terrain = self.scene.add_entity(
            gs.morphs.Terrain(
                n_subterrains=(self.n_levels, self.n_tiles),
                subterrain_size=(self.tile_size, self.tile_size),
                horizontal_scale=0.1,
                subterrain_types=self.terrain_types,
                subterrain_parameters=env_cfg["terrain_parameters"],
            ),
        )

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=(0.0, 0.0, env_cfg["base_init_height"]),
                links_to_keep=env_cfg["foot_names"],
            ),
        )

        # A kinematic Go2 that follows the commanded motion without taking part in the dynamics.
        self.ghost = None
        if ghost:
            self.ghost = self.scene.add_entity(
                gs.morphs.URDF(
                    file="urdf/go2/urdf/go2.urdf",
                    pos=(0.0, 0.0, env_cfg["base_init_height"]),
                ),
                material=gs.materials.Kinematic(),
                surface=gs.surfaces.Default(
                    color=(0.4, 0.7, 1.0),
                    opacity=0.4,
                ),
            )

        self.feet_contact = [
            self.scene.add_sensor(
                gs.sensors.Contact(
                    entity_idx=self.robot.idx,
                    link_idx_local=self.robot.get_link(name).idx_local,
                    draw_debug=show_viewer or record,
                )
            )
            for name in env_cfg["foot_names"]
        ]
        self.base_contact = self.scene.add_sensor(
            gs.sensors.Contact(
                entity_idx=self.robot.idx,
                link_idx_local=self.robot.base_link.idx_local,
            )
        )
        # The rays start just below the trunk and point down, so that they see the ground under and ahead of the robot.
        self.height_scan = self.scene.add_sensor(
            gs.sensors.Raycaster(
                pattern=gs.sensors.GridPattern(
                    resolution=obs_cfg["scan_resolution"],
                    size=obs_cfg["scan_size"],
                ),
                entity_idx=self.robot.idx,
                pos_offset=(0.1, 0.0, -0.08),
                max_range=2.0,
                return_points=False,
                draw_debug=show_viewer or record,
                debug_sphere_radius=0.01,
            )
        )

        self.camera = None
        if record:
            self.camera = self.scene.add_camera(
                res=(1280, 720),
                pos=(-2.5, -2.5, 2.0),
                lookat=(0.0, 0.0, 0.3),
                fov=40,
                debug=True,
            )

        self.scene.build(n_envs=num_envs)

        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dofs_idx_local[0] for name in env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.robot.set_dofs_kp([env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # Domain randomization: one friction ratio for all the links and one payload on the trunk, per environment.
        links_idx_local = range(self.robot.n_links)
        friction_ratio = gs_rand(*env_cfg["friction_ratio_range"], (num_envs, 1)).expand(-1, self.robot.n_links)
        self.robot.set_friction_ratio(friction_ratio, links_idx_local)
        base_idx_local = [self.robot.base_link.idx_local]
        payload = gs_rand(*env_cfg["payload_range"], (num_envs, 1))
        self.robot.set_links_mass(self.robot.get_links_mass(base_idx_local) + payload, base_idx_local)

        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            dtype=gs.tc_float,
            device=gs.device,
        )

        self.base_pos = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.spawn_pos = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.commands = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.heading_target = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.actions = torch.zeros((num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.dof_pos_target = self.default_dof_pos.expand(num_envs, -1).clone()
        self.feet_air_time = torch.zeros((num_envs, len(self.feet_contact)), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.zeros((num_envs,), dtype=gs.tc_int, device=gs.device)
        self.levels = torch.randint(0, env_cfg["max_init_level"] + 1, (num_envs,), device=gs.device)
        self.extras = dict(observations=dict())

        reward_functions = {
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_ang_vel": self._reward_tracking_ang_vel,
            "lin_vel_z": self._reward_lin_vel_z,
            "ang_vel_xy": self._reward_ang_vel_xy,
            "action_rate": self._reward_action_rate,
            "dof_acc": self._reward_dof_acc,
            "torques": self._reward_torques,
            "feet_air_time": self._reward_feet_air_time,
            "similar_to_default": self._reward_similar_to_default,
        }
        self.reward_functions = {name: reward_functions[name] for name in self.reward_scales}
        self.reward_scales = {name: scale * self.dt for name, scale in self.reward_scales.items()}
        self.episode_sums = {
            name: torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device) for name in self.reward_scales
        }

        self.reset()

    def _resample_commands(self, envs_idx):
        n = self.num_envs
        commands = torch.stack(
            (
                gs_rand(*self.command_cfg["lin_vel_x_range"], (n,)),
                gs_rand(*self.command_cfg["lin_vel_y_range"], (n,)),
                torch.zeros((n,), dtype=gs.tc_float, device=gs.device),
            ),
            dim=-1,
        )
        heading_target = gs_rand(-math.pi, math.pi, (n,))
        torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)
        torch.where(envs_idx, heading_target, self.heading_target, out=self.heading_target)

    def _update_curriculum(self, envs_idx):
        distance = torch.linalg.norm(self.base_pos[:, :2] - self.spawn_pos[:, :2], dim=-1)
        commanded_distance = torch.linalg.norm(self.commands[:, :2], dim=-1) * self.env_cfg["episode_length_s"]
        is_level_up = distance > 0.5 * self.tile_size
        is_level_down = ~is_level_up & (distance < 0.25 * commanded_distance)
        levels = self.levels + is_level_up.to(dtype=self.levels.dtype) - is_level_down.to(dtype=self.levels.dtype)
        # An environment that clears the top level restarts from a random one, so the easier levels stay in training.
        levels = torch.where(levels >= self.n_levels, torch.randint_like(levels, 0, self.n_levels), levels)
        torch.where(envs_idx, levels.clamp(min=0), self.levels, out=self.levels)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        # The real robot acts on the previous action, one control period late.
        self.dof_pos_target = self.last_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(self.dof_pos_target, self.motors_dof_idx)
        self.scene.step()

        self.episode_length_buf += 1
        self._update_state()

        # The yaw rate steers toward the heading target.
        heading_error = torch.remainder(self.heading_target - self.heading + math.pi, 2 * math.pi) - math.pi
        self.commands[:, 2] = torch.clip(0.5 * heading_error, *self.command_cfg["ang_vel_range"])

        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        resampling_period = int(self.env_cfg["resampling_time_s"] / self.dt)
        self._resample_commands(self.episode_length_buf % resampling_period == 0)

        # An episode ends when the trunk touches the ground, when the robot falls into a hole, or when it walks off
        # the terrain, the last one counting as a timeout.
        time_outs = self.episode_length_buf > self.max_episode_length
        border = 1.0
        time_outs |= (self.base_pos[:, 0] < border) | (self.base_pos[:, 0] > self.n_levels * self.tile_size - border)
        time_outs |= (self.base_pos[:, 1] < border) | (self.base_pos[:, 1] > self.n_tiles * self.tile_size - border)
        self.reset_buf = time_outs | self.base_contact.read()[:, 0]
        self.reset_buf |= self.base_pos[:, 2] < self.spawn_pos[:, 2] - self.env_cfg["termination_fall_height"]
        self.reset_buf |= self.scene.get_error_envs_mask()
        self.extras["time_outs"] = time_outs.to(dtype=gs.tc_float)

        self._reset_idx(self.reset_buf)
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def _update_state(self):
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.heading = quat_to_xyz(self.base_quat)[:, 2]
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self.torques = self.robot.get_dofs_control_force(self.motors_dof_idx)
        self.feet_in_contact = torch.cat([sensor.read() for sensor in self.feet_contact], dim=-1)
        self.scan = self.height_scan.read().distances

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def _reset_idx(self, envs_idx):
        n_resets = envs_idx.sum()
        self._update_curriculum(envs_idx)

        # Spawn at the center of a random tile of the level, facing a random direction.
        tiles = torch.randint(0, self.n_tiles, (self.num_envs,), device=gs.device)
        spawn_xy = self.tile_size * (torch.stack((self.levels, tiles), dim=-1) + 0.5)
        spawn_z = self.terrain.get_terrain_height(spawn_xy[:, None]).squeeze(-1) + self.env_cfg["base_init_height"]
        spawn_pos = torch.cat((spawn_xy, spawn_z[:, None]), dim=-1)
        torch.where(envs_idx[:, None], spawn_pos, self.spawn_pos, out=self.spawn_pos)
        spawn_euler = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        spawn_euler[:, 2] = gs_rand(-math.pi, math.pi, (self.num_envs,))
        qpos = torch.cat((spawn_pos, xyz_to_quat(spawn_euler), self.default_dof_pos.expand(self.num_envs, -1)), dim=-1)
        self.robot.set_qpos(qpos[envs_idx], envs_idx=envs_idx, zero_velocity=True)

        self.actions.masked_fill_(envs_idx[:, None], 0.0)
        self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
        self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
        self.feet_air_time.masked_fill_(envs_idx[:, None], 0.0)
        self.episode_length_buf.masked_fill_(envs_idx, 0)
        self._update_state()

        self.extras["episode"] = {}
        for name, value in self.episode_sums.items():
            mean = torch.where(n_resets > 0, value[envs_idx].sum() / n_resets, 0.0)
            self.extras["episode"]["rew_" + name] = mean / self.env_cfg["episode_length_s"]
            value.masked_fill_(envs_idx, 0.0)
        self.extras["episode"]["terrain_level"] = self.levels.to(dtype=gs.tc_float).mean()

        self._resample_commands(envs_idx)

    def _update_observation(self):
        scan_obs = torch.clip(self.scan - self.obs_cfg["scan_offset"], -1.0, 1.0) * self.obs_scales["scan"]
        self.obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],
                self.projected_gravity,
                self.commands * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                self.dof_vel * self.obs_scales["dof_vel"],
                self.actions,
                self.feet_in_contact.to(dtype=gs.tc_float),
                scan_obs.reshape((self.num_envs, -1)),
            ),
            dim=-1,
        )

    def reset(self):
        self._reset_idx(torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device))
        self._update_observation()
        return self.get_observations()

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_feet_air_time(self):
        # Reward steps longer than the target air time, counted when a foot lands, and only under a walking command.
        is_first_contact = (self.feet_air_time > 0.0) & self.feet_in_contact
        self.feet_air_time += self.dt
        air_time_target = self.reward_cfg["feet_air_time_target"]
        rew = torch.sum((self.feet_air_time - air_time_target) * is_first_contact, dim=1)
        self.feet_air_time.masked_fill_(self.feet_in_contact, 0.0)
        return rew * (torch.linalg.norm(self.commands[:, :2], dim=1) > 0.1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
