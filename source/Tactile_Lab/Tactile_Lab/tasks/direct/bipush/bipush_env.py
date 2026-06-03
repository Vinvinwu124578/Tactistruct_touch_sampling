# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import time
from attr import set_run_validators
from ipdb import set_trace
import os

import torch
torch.set_printoptions(precision=6, sci_mode=False)
import torch.nn.functional as F

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
# from isaaclab.utils.math import sample_uniform
from .bipush_env_cfg import BiPushEnvCfg 
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors.camera import TiledCamera
from isaaclab.sensors import ContactSensor

import cv2
import numpy as np
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, euler_xyz_from_quat, quat_unique, quat_apply
from isaaclab.utils.math import sample_uniform
from tg3_lab.utility.tactile_sensor import TactileSensor
from tg3_lab.utility.utils import quat_from_rpy, rpy_from_quat

class BiPushEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: BiPushEnvCfg

    def __init__(self, cfg: BiPushEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
        # --------------------------------------------------
        # DOF / action buffers (PER ROBOT)
        # --------------------------------------------------
        self.robot_dof_targets = [
            torch.zeros((self.num_envs, robot.num_joints), device=self.device)
            for robot in self._robots
        ]

        self.prev_targets = [
            torch.zeros((self.num_envs, 7), device=self.device)
            for _ in range(self.robot_number)
        ]

        self.cur_targets = [
            torch.zeros((self.num_envs, 7), device=self.device)
            for _ in range(self.robot_number)
        ]

        # Initialize goal-related variables
        self.current_goal_list_id_all_envs = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )        
        self.current_goal_pose_worldframe_all_envs = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)) # (N, 6)
        self.current_goal_quat_worldframe_all_envs = torch.tensor([0, 0, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)) # (N, 4)
        self.successes_env_id_masks = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.int64)
        # Per-robot cached state (initialised once)
        for name in ["ee_pose_world", "ee_pose_scene", "ee_rpy_work", "ee_pos_work", "ee_quat_work", "ee_pos_b", "ee_quat_b", "ee_linvel", "ee_angvel", "jacobians", "joint_pos"]:
            setattr(self, name, [None] * self.robot_number)
        self.actions = [None] * self.robot_number
        self.pre_actions = [None] * self.robot_number

        # Markers
        if self.cfg.if_debug:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self.obj_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/obj_frame"))
            self.goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/goal_frame"))
            self.ee_markers = []
            for i in range(self.robot_number):
                self.ee_markers.append(
                    VisualizationMarkers(
                        frame_marker_cfg.replace(
                            prim_path=f"/Visuals/ee_robot_{i}"
                        )
                    )
                )

            # frame_marker_cfg = FRAME_MARKER_CFG.copy()
            # frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            # self.camera_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/camera_frame"))

        # #################################### For IK ####################################
        self.diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")  # TODO: set the use_relative_mode for RL training
        self.diff_ik_controller = DifferentialIKController(self.diff_ik_cfg, num_envs=self.num_envs, device=self.device)

        # Define workframe, robot_base_frame, and scene_frame, and initial end-effector pose
        self.workframe_in_scene_frame = torch.tensor(self.cfg.workframe_in_scene_frame).unsqueeze(0).repeat(self.num_envs, 1).to(self.device)
        # self.robot_base_in_scene_frame_0 = torch.tensor(self.cfg.robot_base_in_scene_frame_0).unsqueeze(0).repeat(self.num_envs, 1).to(self.device)
        self.robot_base_in_scene_frame = [
            torch.tensor(self.cfg.robot_base_in_scene_frame_0, device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1),

            torch.tensor(self.cfg.robot_base_in_scene_frame_1, device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1),
        ]

        self.scene_quats = torch.zeros((self.num_envs, 4), device=self.device)
        self.scene_quats[:, 0] = 1.0   # (w, x, y, z), assume the scene frame orientation is aligned with the world frame
        self.workframe_world = self.workframe_in_scene_frame.clone()
        self.workframe_world[:, 0:3] += self.scene.env_origins[:, 0:3]  # assume the scene frame orientation is aligned with the world frame
        # EE initial pose in each robot's own base frame (same value)
        self.ee_init_pose_base = [
            torch.tensor(self.cfg.ee_init_pose_base, device=self.device),
            torch.tensor(self.cfg.ee_init_pose_base, device=self.device),
        ]
        # Create IK command buffers (per robot)
        self.ik_commands = [
            torch.zeros(
                self.num_envs,
                self.diff_ik_controller.action_dim,
                device=self.device,
            )
            for _ in range(self.robot_number)
        ]
        # Initialise IK commands for each robot
        for i in range(self.robot_number):
            self.ik_commands[i][:] = self.ee_init_pose_base[i]

        # --------------------------------------------------
        # Robot entity cfgs (MATCH _setup_scene)
        # --------------------------------------------------
        self.robot_entity_cfgs = []
        for i in range(self.robot_number):
            if self.cfg.robot_name == "franka_panda":
                cfg_i = SceneEntityCfg(
                    f"robot_{i}",
                    joint_names=["panda_joint.*"],
                    body_names=["panda_hand"],
                )
            elif self.cfg.robot_name == "ur5_tactip":
                cfg_i = SceneEntityCfg(
                    f"robot_{i}",
                    joint_names=[".*"],
                    body_names=["tcp_link"],
                )

            cfg_i.resolve(self.scene)
            self.robot_entity_cfgs.append(cfg_i)
        # --------------------------------------------------
        # EE Jacobian indices (per robot)
        # --------------------------------------------------
        self.ee_jacobi_idx = []
        for i, robot in enumerate(self._robots):
            body_id = self.robot_entity_cfgs[i].body_ids[0]
            self.ee_jacobi_idx.append(
                body_id - 1 if robot.is_fixed_base else body_id
            )
        
    def _setup_scene(self):
        self.obs_type = self.cfg.obs_type
        # ------------------
        # Robots
        # ------------------
        self._robots = [
            Articulation(self.cfg.robot_cfg_0),
            Articulation(self.cfg.robot_cfg_1),
        ]
        if self.cfg.robot_num != len(self._robots):
            assert False, f"robot_num {self.cfg.robot_num} does not match the number of robot configs provided {len(self._robots)}"
        else:
            self.robot_number = self.cfg.robot_num
            
        for i, robot in enumerate(self._robots):
            self.scene.articulations[f"robot_{i}"] = robot
        # keep explicit handles if you still want them
        self._robot_0, self._robot_1 = self._robots
        # ------------------
        # Object
        # ------------------
        self._object = RigidObject(self.cfg.object_cfg)
        self.scene.rigid_objects["object"] = self._object
        # ------------------
        # Goals
        # ------------------
        self._goals = []
        for i, goal_cfg in enumerate(self.cfg.goal_cfgs):
            goal = RigidObject(goal_cfg)
            self.scene.rigid_objects[f"goal_{i}"] = goal
            self._goals.append(goal)
        # ------------------
        # Contact sensors (one per robot)
        # ------------------
        self._contact_sensors = [
            ContactSensor(self.cfg.contact_sensor_cfg_0),
            ContactSensor(self.cfg.contact_sensor_cfg_1),
        ]
        for i, sensor in enumerate(self._contact_sensors):
            self.scene.sensors[f"contact_sensor_{i}"] = sensor
        self._contact_sensor_0, self._contact_sensor_1 = self._contact_sensors
        # ------------------
        # Tactile cameras
        # ------------------
        if self.obs_type == "tactile":
            self.tactile_sensors = {}
            self._if_loaded_tactile_reference_image = {}
            for idx, cfg in enumerate(self.cfg.tactile_sensor_cfgs):
                # Create camera
                cam = TiledCamera(cfg.tactile_camera)
                self.scene.sensors[f"tactile_camera_{idx}"] = cam
                # Create tactile sensor wrapper
                self.tactile_sensors[idx] = TactileSensor(
                    cam,
                    cfg,
                    self.device,
                    self.num_envs,
                    __file__,
                )
                # Reference image init flag
                self._if_loaded_tactile_reference_image[idx] = False
        # ------------------
        # Terrain
        # ------------------
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # ------------------
        # Cached tensors
        # ------------------
        self.tcp_lims = [
            torch.tensor(self.cfg.tcp_lims_robot_0, device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1, 1),

            torch.tensor(self.cfg.tcp_lims_robot_1, device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1, 1),
        ]
        
        self.robot_ee_speed_scale = torch.tensor(
            self.cfg.robot_ee_speed_scale, device=self.device
        )

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def init_goals(self, env_ids):
        # traj_orn_workframe_all_envs = torch.zeros((self.cfg.traj_n_points, 4), dtype=torch.float, device=self.device)
        self.current_goal_list_id_all_envs[env_ids] = -1
        if self.cfg.traj_type == "straight":
            traj_pos_workframe_all_envs, traj_rpy_workframe_all_envs = self.update_trajectory_straight()
        elif self.cfg.traj_type == "random_curved":
            traj_pos_workframe_all_envs, traj_rpy_workframe_all_envs = self.update_trajectory_random_curved()

        traj_quat_workframe_all_envs = quat_from_rpy(traj_rpy_workframe_all_envs)
        # traj_pos_workframe_all_envs = torch.cat([traj_pos_workframe_all_envs, traj_quat_workframe_all_envs], dim=-1)
        N, T, _ = traj_pos_workframe_all_envs.shape  # N for num_envs, T for traj_n_points
        # expand workframe pose to [N, T, *]
        workframe_pos_worldframe = self.workframe_world[:, None, 0:3].expand(N, T, 3)
        workframe_quat_worldframe = self.workframe_world[:, None, 3:7].expand(N, T, 4)

        goals_pos_in_worldframe, goals_quat_in_worldframe = combine_frame_transforms(
            workframe_pos_worldframe,                 # [N, T, 3]
            workframe_quat_worldframe,                # [N, T, 4]
            traj_pos_workframe_all_envs,      # [N, T, 3]
            traj_quat_workframe_all_envs,     # [N, T, 4]
        )
        goals_pose_in_worldframe = torch.cat([goals_pos_in_worldframe, goals_quat_in_worldframe], dim=-1)
        for i, goal in enumerate(self._goals):
            goal.write_root_pose_to_sim(goals_pose_in_worldframe[env_ids, i, :7], env_ids)
            goal.reset(env_ids)
        # tip_rpy_deg = torch.rad2deg(traj_rpy_workframe[0][2])
        # print("tip_rpy:", tip_rpy_deg)

    def update_trajectory_random_curved(self):
        """
        Torch-only smooth noisy trajectory.
        """
        device = self.device
        num_envs = self.num_envs
        n_pts = self.cfg.traj_n_points

        traj_pos_workframe_all_envs = torch.zeros(
            (num_envs, n_pts, 3), device=device
        )
        traj_rpy_workframe_all_envs = torch.zeros(
            (num_envs, n_pts, 3), device=device
        )

        # base params
        init_offset = self.cfg.object_size / 2 + self.cfg.traj_spacing
        dist = torch.arange(n_pts, device=device, dtype=torch.float)

        # random heading per env
        traj_ang = torch.rand(num_envs, device=device) * (torch.pi / 4) - (torch.pi / 8)
        dir_x = -torch.sin(traj_ang)
        dir_y = torch.cos(traj_ang)

        # ---- smooth noise generation ----
        noise = torch.randn(num_envs, n_pts, device=device)

        # 1D smoothing kernel (Gaussian-like)
        kernel_size = 9
        sigma = 2.0
        t = torch.arange(kernel_size, device=device) - kernel_size // 2
        kernel = torch.exp(-0.5 * (t / sigma) ** 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1)

        noise = noise.unsqueeze(1)                       # [N,1,T]
        noise = F.conv1d(noise, kernel, padding=kernel_size // 2)
        noise = noise.squeeze(1)                         # [N,T]

        noise = noise * self.cfg.traj_max_perturb
        noise = noise - noise[:, :1]                     # zero initial offset

        # ---- build trajectory ----
        dist = dist * self.cfg.traj_spacing

        x = dist[None, :] * dir_x[:, None] + noise * dir_y[:, None]
        y = init_offset + dist[None, :] * dir_y[:, None] - noise * dir_x[:, None]
        z = torch.full((num_envs, n_pts), self.cfg.goal_z_pos, device=device)

        traj_pos_workframe_all_envs[:] = torch.stack([x, y, z], dim=-1)

        # ---- orientation ----
        dx = torch.gradient(x, dim=1)[0]
        dy = torch.gradient(y, dim=1)[0]
        traj_rpy_workframe_all_envs[:, :, 2] = torch.atan2(dy, dx) - torch.pi / 2

        return traj_pos_workframe_all_envs, traj_rpy_workframe_all_envs

    def update_trajectory_straight(self,):
        """
        traj_pos_workframe: Tensor [num_envs, traj_n_points, 3]
        """
        traj_pos_workframe_all_envs = torch.zeros(
            (self.num_envs, self.cfg.traj_n_points, 3),
            dtype=torch.float,
            device=self.device,
        )
        traj_rpy_workframe_all_envs = torch.zeros(
            (self.num_envs, self.cfg.traj_n_points, 3),
            dtype=torch.float,
            device=self.device,
        )
        n_pts = self.cfg.traj_n_points
        device = self.device
        # (num_envs,) random angle per env
        traj_ang = torch.rand(self.num_envs, device=device) * (torch.pi / 4) - (torch.pi / 8)
        # traj_ang = torch.rand(self.num_envs, device=device) * 0
        # scalar offset
        init_offset = self.cfg.object_size / 2 + self.cfg.traj_spacing
        # (traj_n_points,)
        dist = torch.arange(n_pts, device=device, dtype=torch.float) * self.cfg.traj_spacing
        # directions: (num_envs,)
        dir_x = torch.cos(traj_ang)
        dir_y = torch.sin(traj_ang)
        # broadcast to (num_envs, traj_n_points)
        x = init_offset + dist[None, :] * dir_x[:, None]
        y = dist[None, :] * dir_y[:, None]
        z = torch.full((self.num_envs, n_pts), self.cfg.goal_z_pos, device=device)
        # stack → (num_envs, traj_n_points, 3)
        traj_pos_workframe_all_envs[:] = torch.stack([x, y, z], dim=-1)
        # orientation (yaw) along trajectory
        x = traj_pos_workframe_all_envs[:, :, 0]   # [num_envs, T]
        y = traj_pos_workframe_all_envs[:, :, 1]   # [num_envs, T]
        dx = torch.gradient(x, dim=1)[0]
        dy = torch.gradient(y, dim=1)[0]
        traj_rpy_workframe_all_envs[:, :, 2] = (
            torch.atan2(dy, dx)
        )
        return traj_pos_workframe_all_envs, traj_rpy_workframe_all_envs

    def check_TCP_vel_lims(self, robot_idx: int, vels: torch.Tensor):  # TODO: move this to a utils file
        """
        Check whether action will take TCP outside of limits for ONE robot,
        zero any velocities that will.

        Args:
            robot_idx: index of robot (0 or 1)
            vels: [num_envs, 6] TCP velocities
        """
        # EE pose in work frame (per robot)
        ee_rpy_work = rpy_from_quat(self.ee_quat_work[robot_idx])
        ee_pose_work = torch.cat(
            (self.ee_pos_work[robot_idx], ee_rpy_work),
            dim=-1,   # (num_envs, 6)
        )
        # TCP limits for this robot
        llims = self.tcp_lims[robot_idx][:, :, 0]   # (num_envs, 6)
        ulims = self.tcp_lims[robot_idx][:, :, 1]   # (num_envs, 6)
        # Identify violations based on current position and velocity direction
        exceed_llims = torch.logical_and(ee_pose_work < llims, vels < 0)
        exceed_ulims = torch.logical_and(ee_pose_work > ulims, vels > 0)
        exceeded = torch.logical_or(exceed_llims, exceed_ulims)  # (num_envs, 6)
        # Cap velocities
        capped_vels = vels.clone()
        capped_vels[exceeded] = 0.0
        return capped_vels

    # pre-physics step calls
    def compute_target_pose(self, pre_targets_pose: torch.Tensor, actions: torch.Tensor):
        """
        Update the target pose (quaternion) based on the actions (rpy) and previous target pose (quaternion).
        """
        pre_targets_rpy = rpy_from_quat(pre_targets_pose[:, 3:7])
        targets_rpy = pre_targets_rpy + actions[:, 3:6]
        target_quat = quat_from_rpy(targets_rpy)
        target_pos = pre_targets_pose[:, 0:3] + actions[:, 0:3]
        target_pose = torch.cat([target_pos, target_quat], dim=-1)

        return target_pose

    def transform_action_to_tcp_frame(self, robot_idx: int, actions: torch.Tensor):
        """
        Transform actions from TCP local frame to world frame for ONE robot.

        Args:
            robot_idx: which robot (0 or 1)
            actions: [num_envs, 6]
        """
        # local TCP basis vectors
        perp_vector = torch.tensor(
            [1.0, 0.0, 0.0], device=self.device
        ).expand(self.num_envs, -1)   # perpendicular
        par_vector = torch.tensor(
            [0.0, 1.0, 0.0], device=self.device
        ).expand(self.num_envs, -1)   # parallel (outwards)
        # rotate local vectors into world frame using THIS robot's TCP orientation
        perp_tip_dir_world = quat_apply(
            self.ee_pose_scene[robot_idx][:, 3:7], perp_vector
        )  # [B, 3]
        par_tip_dir_world = quat_apply(
            self.ee_pose_scene[robot_idx][:, 3:7], par_vector
        )  # [B, 3]
        # scales from actions
        perp_scale = actions[:, 0]   # [B]
        par_scale = actions[:, 1]    # [B]
        # scale direction vectors
        perp_action_world = perp_tip_dir_world * perp_scale.unsqueeze(-1)
        par_action_world = par_tip_dir_world * par_scale.unsqueeze(-1)
        # accumulate XY components
        total_xy = perp_action_world[:, :2] + par_action_world[:, :2]  # [B, 2]
        actions[:, 0] = total_xy[:, 0]
        actions[:, 1] = total_xy[:, 1]
        return actions

    def _expand_action_3d_to_6d(self, actions_3d: torch.Tensor):  # TODO: move this to a utils file
        # 3D action: [N, 3] (dx, dy, dtheta)
        # pad to 6D
        actions_6d = F.pad(actions_3d, pad=(0, 3), mode="constant", value=0)  # [N, 6]
        # swap z and pitch axes (your existing hack)
        actions_6d[:, [2, 5]] = actions_6d[:, [5, 2]]
        # actions_6d[:, [0, 1]] = actions_6d[:, [1, 0]]
        # push along x axis (your existing behaviour)
        actions_6d[:, 0] += 1.0

        return actions_6d

    def _pre_physics_step(self, actions: torch.Tensor):
        # actions[:,0] = 1.0
        for robot_idx, robot in enumerate(self._robots):
            entity_cfg = self.robot_entity_cfgs[robot_idx]
            ee_jacobi_idx = self.ee_jacobi_idx[robot_idx]
            # Obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[
                :, ee_jacobi_idx, :, entity_cfg.joint_ids
            ]
            robot_root_pose_world = robot.data.root_pose_w.clone()
            joint_pos = robot.data.joint_pos[:, entity_cfg.joint_ids].clone()
            # print("joint_pos:", joint_pos) #  For getting the initial joint positions after determine the ee init pose in base frame
            ee_pose_world = robot.data.body_pose_w[
                :, entity_cfg.body_ids[0]
            ].clone()
            # EE pose in WORK frame (shared workframe)
            ee_pos_work, ee_quat_work = subtract_frame_transforms(
                self.workframe_world[:, 0:3],
                self.workframe_world[:, 3:7],
                ee_pose_world[:, 0:3],
                ee_pose_world[:, 3:7],
            )
            # EE pose in ROBOT BASE frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                robot_root_pose_world[:, 0:3],
                robot_root_pose_world[:, 3:7],
                ee_pose_world[:, 0:3],
                ee_pose_world[:, 3:7],
            )
            # Cache per-robot values
            self.ee_pos_work[robot_idx] = ee_pos_work
            self.ee_quat_work[robot_idx] = ee_quat_work
            self.ee_pos_b[robot_idx] = ee_pos_b
            self.ee_quat_b[robot_idx] = ee_quat_b

            # (optional) cache jacobian / joint_pos if used later
            self.jacobians[robot_idx] = jacobian
            self.joint_pos[robot_idx] = joint_pos
        actions_r0 = actions[:, 0:int(self.cfg.action_space/2)]
        actions_r1 = actions[:, int(self.cfg.action_space/2):self.cfg.action_space]
        actions_6d = []
        if not hasattr(self, "actions"):
            self.actions = [None] * self.robot_number
            self.pre_actions = [None] * self.robot_number

        for robot_idx, act_3d in enumerate([actions_r0, actions_r1]):
            act_6d = self._expand_action_3d_to_6d(act_3d)
            # previous actions
            if self.actions[robot_idx] is not None:
                self.pre_actions[robot_idx] = self.actions[robot_idx].clone()
                if self.pre_actions[robot_idx].shape[1] == 2:
                    self.pre_actions[robot_idx] = F.pad(
                        self.pre_actions[robot_idx],
                        pad=(0, 4),
                        mode="constant",
                        value=0,
                    )  # [N, 6]
            else:
                self.pre_actions[robot_idx] = torch.zeros(
                    (self.num_envs, 6),
                    device=self.device,
                )
            # clamp
            act_6d = act_6d.clamp(-self.cfg.clip_actions, self.cfg.clip_actions)
            # TCP frame
            tcp_actions = self.transform_action_to_tcp_frame(robot_idx, act_6d)
            # scale
            scaled_actions = (
                self.robot_ee_speed_scale
                * self.dt
                * tcp_actions
                * self.cfg.action_scale
            )
            # TCP limit check (per robot)
            checked_scaled_actions = self.check_TCP_vel_lims(robot_idx, scaled_actions)
            # integrate target
            # set_trace()
            cur_targets = self.compute_target_pose(
                self.prev_targets[robot_idx], checked_scaled_actions
            )
            # cache
            self.prev_targets[robot_idx][:] = cur_targets
            self.ik_commands[robot_idx] = cur_targets
            actions_6d.append(act_6d)

        for robot_idx, robot in enumerate(self._robots):
            self.diff_ik_controller.set_command(self.ik_commands[robot_idx])
            self.robot_dof_targets[robot_idx] = self.diff_ik_controller.compute(
                self.ee_pos_b[robot_idx],
                self.ee_quat_b[robot_idx],
                self.jacobians[robot_idx],
                self.joint_pos[robot_idx],
            )

    def _apply_action(self):
        for robot_idx, robot in enumerate(self._robots):
            robot.set_joint_position_target(
                self.robot_dof_targets[robot_idx],
                joint_ids=self.robot_entity_cfgs[robot_idx].joint_ids,
            )

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()  # TODO: check whether to put this in get_done or get_reward!!!
        # any_tip_no_contact = torch.zeros(
        #     self.num_envs, device=self.device, dtype=torch.bool
        # )

        # for robot_idx in range(self.robot_number):
        #     any_tip_no_contact |= self.no_contact_tip_mask[robot_idx].squeeze(-1)

        # terminated = self.successes_env_id_masks.bool() | any_tip_no_contact
        terminated = self.successes_env_id_masks.bool()
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    def cos_tcp_dist_to_obj(self):
        """
        Calculate the cosine distance between the TCP and the object.
        This is calculated as the dot product of the TCP orientation and the object orientation.
        """
        cur_obj_orn_worldframe = self._object.data.root_quat_w
        # get normal vector of object
        q_obj = cur_obj_orn_worldframe
        init_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device).repeat(q_obj.shape[0], 1)
        obj_vector = quat_apply(q_obj, init_vec)   # [B,3]

        cos_dist_robots = []
        # loop over robots
        for robot_idx in range(self.robot_number):
            # tcp orientation (world frame)
            cur_tcp_orn_worldframe = self.ee_pose_world[robot_idx][:, 3:7]
            q_tip = cur_tcp_orn_worldframe
            tip_init_vector = init_vec
            tip_vector = quat_apply(q_tip, tip_init_vector)   # [N, 3]
            # cosine similarity
            cos_sim = (obj_vector * tip_vector).sum(dim=-1) / (
                obj_vector.norm(dim=-1) * tip_vector.norm(dim=-1)
            )
            # cosine distance
            cos_dist = 1.0 - cos_sim
            cos_dist_robots.append(cos_dist)

        # average over robots (same as PyBullet implementation)
        cos_dist = torch.stack(cos_dist_robots, dim=0).mean(dim=0)
        return cos_dist

    def orn_obj_dist_to_goal(self):
        """ Calculate the orientation distance from the object to the goal.
        This is calculated as the angle between the quaternion of the goal and the quaternion of the object.
        """
        cur_obj_orn_worldframe = self._object.data.root_quat_w
        # q_g, q_o: [B, 4]
        q_g = self.current_goal_quat_worldframe_all_envs          # [15, 4]
        q_o = cur_obj_orn_worldframe                           # [15, 4]
        # (optional) ensure unit quaternions
        q_g = q_g / q_g.norm(dim=-1, keepdim=True)
        q_o = q_o / q_o.norm(dim=-1, keepdim=True)
        # batched dot: [B]
        dots = (q_g * q_o).sum(dim=-1)
        # distance (two equivalent forms—pick one)
        dist = torch.acos(torch.clamp(2 * (dots ** 2) - 1, -1.0, 1.0))   # [B]
        # print("orn_dist:", dist)
        return dist
    
    def xy_obj_dist_to_tip(self):
        """
        Calculate the distance from the object to the tip in the xy plane.
        """
        # Get the tip position in workframe
        tip_xy_pos_work = self.ee_pos_work[:, :2]
        # Get the object position in workframe
        obj_xy_pos_work = self.obj_pos_work[:, :2]
        # Calculate the distance in the xy plane
        xy_dist = torch.norm(tip_xy_pos_work - obj_xy_pos_work, dim=1)
        # print("xy_dist:", xy_dist)
        return xy_dist
    
    def xy_obj_dist_to_goal(self):
        """
        Calculate the distance from the object to the goal in the xy plane.
        """
        # Get the goal position in workframe
        goal_pos_xy_work = self.current_goal_pos_work[:, :2]
        # Get the object x,y position in workframe
        obj_pos_xy_work = self.obj_pos_work[:, :2]
        # Calculate the distance in the xy plane
        xy_dist = torch.norm(goal_pos_xy_work - obj_pos_xy_work, dim=1)
        return xy_dist
    
    def _compute_intermediate_values(self):
        
        # data for robot
        for robot_idx, robot in enumerate(self._robots):
            entity_cfg = self.robot_entity_cfgs[robot_idx]
            # EE pose in WORLD frame
            ee_pose_world = robot.data.body_pose_w[
                :, entity_cfg.body_ids[0]
            ].clone()
            self.ee_pose_world[robot_idx] = ee_pose_world
            # EE pose in SCENE frame
            ee_pos_scene, ee_quat_scene = subtract_frame_transforms(
                self.scene.env_origins,
                self.scene_quats,
                ee_pose_world[:, 0:3],
                ee_pose_world[:, 3:7],
            )
            self.ee_pose_scene[robot_idx] = torch.cat(
                [ee_pos_scene, ee_quat_scene], dim=-1
            )
            # EE pose in WORK frame (shared workframe)
            ee_pos_work, ee_quat_work = subtract_frame_transforms(
                self.workframe_world[:, 0:3],
                self.workframe_world[:, 3:7],
                ee_pose_world[:, 0:3],
                ee_pose_world[:, 3:7],
            )
            self.ee_pos_work[robot_idx] = ee_pos_work
            self.ee_quat_work[robot_idx] = ee_quat_work
            self.ee_rpy_work[robot_idx] = rpy_from_quat(ee_quat_work)
            # EE velocities in WORLD frame
            self.ee_linvel[robot_idx] = robot.data.body_lin_vel_w[
                :, entity_cfg.body_ids[0]
            ].clone()
            self.ee_angvel[robot_idx] = robot.data.body_ang_vel_w[
                :, entity_cfg.body_ids[0]
            ].clone()
        # data for object
        # self.obj_pos = self._object.data.root_pos_w - self.scene.env_origins
        self.obj_pos_work, self.obj_quat_work = subtract_frame_transforms(
            self.workframe_world[:, 0:3], self.workframe_world[:, 3:7],
            self._object.data.root_pos_w,
            self._object.data.root_quat_w,
        )
        self.obj_rpy_work = rpy_from_quat(self.obj_quat_work)
        self.obj_velocities = self._object.data.root_vel_w
        self.obj_linvel = self._object.data.root_lin_vel_w
        self.obj_angvel = self._object.data.root_ang_vel_w
        # compute distances
        self.obj_goal_pos_dist = self.xy_obj_dist_to_goal()
        self.obj_goal_orn_dist = self.orn_obj_dist_to_goal()
        self.tip_obj_orn_dist = self.cos_tcp_dist_to_obj()
        ###################################### data for goals ####################################
        self.goal_update(self.get_sub_goal_update_env_ids())
        ####################################contact sensor####################################
        # per-robot contact force and mask
        # per-robot contact force and mask
        self.tip_contact_force = []
        self.no_contact_tip_mask = []
        # gate threshold
        contact_check_after_steps = 100
        for robot_idx in range(self.robot_number):
            tip_contact_force = self._contact_sensors[robot_idx].data.net_forces_w
            # tip_contact_force: [num_envs, 3] or [num_envs, 1, 3]
            tip_force_mag = tip_contact_force.norm(dim=-1).squeeze(-1)  # [num_envs]
            # raw no-contact
            raw_no_contact = tip_force_mag <= 1e-6  # [num_envs]
            # only enable no-contact AFTER 100 steps
            no_contact_tip_mask = raw_no_contact & (
                self.episode_length_buf >= contact_check_after_steps
            )
            self.tip_contact_force.append(tip_contact_force)
            self.no_contact_tip_mask.append(no_contact_tip_mask)


    def get_sub_goal_update_env_ids(self):
        sub_goal_update_mask = self.obj_goal_pos_dist < self.cfg.sub_goal_threshold
        goal_update_env_ids = torch.nonzero(sub_goal_update_mask, as_tuple=False).squeeze(-1)
        return goal_update_env_ids

    def goal_update(self, update_goal_env_ids):
        # torch.cuda.synchronize()
        # t0 = time.perf_counter()

        device = self.device
        update_goal_env_ids = torch.as_tensor(update_goal_env_ids, device=device)

        # increment goal index
        self.current_goal_list_id_all_envs[update_goal_env_ids] += 1
        updated_ids = self.current_goal_list_id_all_envs[update_goal_env_ids]

        # check termination
        active_mask = updated_ids < self.cfg.traj_n_points
        active_env_ids = update_goal_env_ids[active_mask]
        achieved_subgoal_ids = updated_ids[active_mask] - 1

        finished_env_ids = update_goal_env_ids[~active_mask]
        self.successes_env_id_masks = torch.isin(self.env_ids_tensor, finished_env_ids).float()

        if active_env_ids.numel() == 0:
            return
        
        active_subgoal_ids = self.current_goal_list_id_all_envs[active_env_ids]
        achieved_subgoal_ids = updated_ids[active_mask] - 1

        for subgoal_id in range(self.cfg.traj_n_points):
            # ---- achieved sub-goal (visualization) ----
            mask_achieved = achieved_subgoal_ids == subgoal_id
            env_ids_achieved = active_env_ids[mask_achieved]

            if env_ids_achieved.numel() > 0:
                goal = self._goals[subgoal_id]

                pos = goal.data.root_pos_w[env_ids_achieved]
                quat = goal.data.root_quat_w[env_ids_achieved]

                pos = pos.clone()
                pos[:, 2] += 0.05  # z += 1

                goal.write_root_pose_to_sim(
                    torch.cat([pos, quat], dim=1),
                    env_ids_achieved
                )

            # ---- current sub-goal (buffers) ----
            mask_current = active_subgoal_ids == subgoal_id
            env_ids_current = active_env_ids[mask_current]

            if env_ids_current.numel() > 0:
                goal = self._goals[subgoal_id]

                pos = goal.data.root_pos_w[env_ids_current]
                quat = goal.data.root_quat_w[env_ids_current]
                rpy = rpy_from_quat(quat)

                self.current_goal_quat_worldframe_all_envs[env_ids_current] = quat
                self.current_goal_pose_worldframe_all_envs[env_ids_current] = torch.cat(
                    [pos, rpy], dim=1
                )

        self.current_goal_pos_work , self.current_goal_quat_work = subtract_frame_transforms(
            self.workframe_world[:, 0:3], self.workframe_world[:, 3:7],
            self.current_goal_pose_worldframe_all_envs[:, 0:3],
            self.current_goal_quat_worldframe_all_envs,)

        self.current_goal_rpy_work = rpy_from_quat(self.current_goal_quat_work)

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        return self._compute_rewards(
            self.obj_goal_pos_dist,
            self.obj_goal_orn_dist,
            self.tip_obj_orn_dist,
            # self.obj_tip_pos_dist,
            self.cfg.obj_goal_pos_dist_scale,
            self.cfg.obj_goal_orn_dist_scale,
            self.cfg.tip_obj_orn_dist_scale,
            # self.cfg.obj_tip_pos_dist_scale,
            self.successes_env_id_masks,
            # self.no_contact_tip_mask,
            # self.too_far_mask,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        ################################ Reset robot ###############################################
        for robot_idx, robot in enumerate(self._robots):
            # Reset joints to default
            joint_pos_env_ids = robot.data.default_joint_pos[env_ids].clone()
            joint_vel_env_ids = robot.data.default_joint_vel[env_ids].clone()
            robot.set_joint_position_target(joint_pos_env_ids, env_ids=env_ids)
            robot.write_joint_state_to_sim(joint_pos_env_ids, joint_vel_env_ids, env_ids=env_ids)
            robot.reset(env_ids)
            # Reset IK commands (base frame)
            self.ik_commands[robot_idx][env_ids, :] = self.ee_init_pose_base[robot_idx].clone()
            self.prev_targets[robot_idx][env_ids, :] = self.ee_init_pose_base[robot_idx].clone()
            self.cur_targets[robot_idx][env_ids, :] = self.ee_init_pose_base[robot_idx].clone()

        ## Now reset it to the ee_init_pose_base! Question: is the ik_commands refer to world frame or scene frame or robot base frame? It is the base frame!
        ############################# reset actions and ik commands #############################
        # reset controller
        self.diff_ik_controller.reset(env_ids)
        # set IK commands for all robots
        for robot_idx in range(self.robot_number):
            self.diff_ik_controller.set_command(self.ik_commands[robot_idx])

        # obtain quantities from simulation
        ################################ Apply IK to robots #########################################
        for robot_idx, robot in enumerate(self._robots):
            entity_cfg = self.robot_entity_cfgs[robot_idx]
            ee_jacobi_idx = self.ee_jacobi_idx[robot_idx]
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[
                :, ee_jacobi_idx, :, entity_cfg.joint_ids
            ]
            ee_pose_world = robot.data.body_pose_w[:, entity_cfg.body_ids[0]].clone()
            robot_root_pose_world = robot.data.root_pose_w.clone()
            joint_pos = robot.data.joint_pos[:, entity_cfg.joint_ids].clone()
            # EE pose in robot base frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                robot_root_pose_world[:, 0:3],
                robot_root_pose_world[:, 3:7],
                ee_pose_world[:, 0:3],
                ee_pose_world[:, 3:7],
            )
            # compute IK
            robot_dof_targets = self.diff_ik_controller.compute(
                ee_pos_b, ee_quat_b, jacobian, joint_pos
            )
            joint_vel = robot.data.default_joint_vel

            robot.set_joint_position_target(
                robot_dof_targets[env_ids, :], env_ids=env_ids
            )
            robot.write_joint_state_to_sim(
                robot_dof_targets[env_ids, :],
                joint_vel[env_ids, :],
                env_ids=env_ids,
            )
            robot.reset(env_ids)
        ####################### reset object#########################
        obj_default_state_env_ids_scene = self._object.data.default_root_state.clone()[env_ids]
        # Convert state in scene frame to world frame
        obj_default_state_env_ids_world = obj_default_state_env_ids_scene.clone()
        obj_default_state_env_ids_world[:, 0:3] = (
            obj_default_state_env_ids_scene[:, 0:3] + self.scene.env_origins[env_ids]
        )
        # pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        # # global object positions
        # obj_default_state[:, 0:3] = (
        #     obj_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        # )

        # rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        # obj_default_state[:, 3:7] = randomize_rotation(
        #     rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        # )
        obj_default_state_env_ids_world[:, 7:] = torch.zeros_like(self._object.data.default_root_state[env_ids, 7:]) # Set all the velocities to zero
        self._object.write_root_pose_to_sim(obj_default_state_env_ids_world[:, :7], env_ids)
        self._object.write_root_velocity_to_sim(obj_default_state_env_ids_world[:, 7:], env_ids)

        ####################### reset goals#########################
        for goal in self._goals:
            # Reset the goal pose to default pose with zero velocity
            goal_default_state = goal.data.default_root_state.clone()[env_ids]
            goal_default_state[:, 7:] = torch.zeros_like(goal.data.default_root_state[env_ids, 7:])
            goal.write_root_pose_to_sim(goal_default_state[:, :7], env_ids)
            goal.write_root_velocity_to_sim(goal_default_state[:, 7:], env_ids)

        # reset goal-related variables
        self.successes_env_id_masks[env_ids] = 0
        # reset goals
        self.init_goals(env_ids)
        self.goal_update(env_ids)
        # compute intermediate values after reset
        self._compute_intermediate_values()

    def _get_oracle_obs(self) -> dict[str, torch.Tensor]:
        """
        Get the oracle observations. All in world frame. #TODO: Change to workframe when deploy to real robot
        """
        # Get the goal position in workframe
        current_goal_pos_xy_work = self.current_goal_pos_work[:, :2]
        goal_yaw = self.current_goal_rpy_work[:, 2]
        # Get the object x,y position in workframe
        obj_pos_xy_work = self.obj_pos_work[:, :2]
        obj_yaw = self.obj_rpy_work[:, 2]
        shared_obs = torch.cat(
            (
                current_goal_pos_xy_work,
                goal_yaw.unsqueeze(-1),
                obj_pos_xy_work,
                obj_yaw.unsqueeze(-1),
                self.obj_linvel[:, :2],
                self.obj_angvel[:, 2].unsqueeze(-1),
            ),
            dim=-1,
        )
        robot_obs = self.get_robots_obs()
        oracle_obs = torch.cat(
            robot_obs + [shared_obs],
            dim=-1,
        )
        return oracle_obs

    def get_robots_obs(self) -> torch.Tensor:
        robot_obs = []
        for robot_idx in range(self.robot_number):
            obs_i = torch.cat(
                (
                    self.ee_pos_work[robot_idx][:, :2],
                    self.ee_rpy_work[robot_idx][:, 2].unsqueeze(-1),
                    self.ee_linvel[robot_idx][:, :2],
                    self.ee_angvel[robot_idx][:, 2].unsqueeze(-1),
                ),
                dim=-1,
            )
            robot_obs.append(obs_i)
        return robot_obs

    def get_proprioception_and_goal_for_obs_dict(self) -> torch.Tensor:
        # Get the goal position in workframe
        current_goal_pos_xy_work = self.current_goal_pos_work[:, :2]
        goal_yaw = self.current_goal_rpy_work[:, 2]
        shared_obs = torch.cat(
            (
                current_goal_pos_xy_work,
                goal_yaw.unsqueeze(-1),
            ),
            dim=-1,
        )
        robot_obs = self.get_robots_obs()

        proprioception_and_goal = torch.cat(
            robot_obs + [shared_obs],
            dim=-1,
        )
        return proprioception_and_goal

    def _get_observations(self) -> dict:
        if self.cfg.if_debug:
            for robot_idx in range(self.robot_number):
                ee_pose_w = self.ee_pose_world[robot_idx]   # [N, 7]
                self.ee_markers[robot_idx].visualize(
                    ee_pose_w[:, 0:3],
                    ee_pose_w[:, 3:7],
                )
            self.obj_marker.visualize(self._object.data.root_pos_w, self._object.data.root_quat_w)
            self.goal_marker.visualize(
                self.current_goal_pose_worldframe_all_envs[:, 0:3],
                self.current_goal_quat_worldframe_all_envs,
            )
        if self.obs_type == "oracle":
            obs = self._get_oracle_obs()
            return {"oracle": torch.clamp(obs, -5.0, 5.0)}  # TODO: Finetune the clamp value (for oracle, max=1, min=-1 due to positional encoding. For tactile image? ). This clamping is for numerical stability
        elif self.obs_type == "tactile":
            imgs = []
            for i, ts in self.tactile_sensors.items():
                if self.cfg.tactile_sensor_cfgs[i].if_save_tactile_reference_image:
                    ts._save_tactile_reference_images()
                if not self._if_loaded_tactile_reference_image[i]:
                    self._if_loaded_tactile_reference_image[i] = True
                    ts._load_tactile_reference_images()
                tactile_obs = ts._get_tactile_images_tensors()    # [B, C, H, W], for rsl-rl
                imgs.append(tactile_obs)
            # concatenate along WIDTH
            img = torch.cat(imgs, dim=3)                         # [B, C, H, 2W]
            img = torch.clamp(img, 0.0, 1.0)
            feature_obs = self.get_proprioception_and_goal_for_obs_dict()
            return {
                "image": img,          # CNN input (B, C, H, W)
                "feature": feature_obs   # MLP input (B, 6)
            }

    def _compute_rewards(
        self,
        # constant_reward: torch.Tensor,
        obj_goal_pos_dist: torch.Tensor,
        obj_goal_orn_dist: torch.Tensor,
        tip_obj_orn_dist: torch.Tensor,
        # obj_tip_pos_dist: torch.Tensor,
        obj_goal_pos_dist_scale: float,
        obj_goal_orn_dist_scale: float,
        tip_obj_orn_dist_scale: float,
        # obj_tip_pos_dist_scale: float,
        successes_env_id_masks: torch.Tensor,
        # too_far_mask: torch.Tensor,
        # no_contact_tip_mask: list,
    ):
        contact_detection_off = False

        # Reward for reaching the goal position
        reward_obj_goal_pos = -obj_goal_pos_dist_scale * obj_goal_pos_dist
        # Reward for reaching the goal orientation
        reward_obj_goal_orn = -obj_goal_orn_dist_scale * obj_goal_orn_dist
        # Reward for aligning the tip orientation with the object orientation
        reward_tip_obj_orn = -tip_obj_orn_dist_scale * tip_obj_orn_dist
        # Reward for bringing the tip close to the object
        # reward_obj_tip_pos = -obj_tip_pos_dist_scale * obj_tip_pos_dist

        rewards = (
            reward_obj_goal_pos
            + reward_obj_goal_orn
            + reward_tip_obj_orn
            # + reward_obj_tip_pos
        )

        # if not contact_detection_off:
        #     for robot_idx in range(self.robot_number):
        #         no_contact_mask = no_contact_tip_mask[robot_idx]
        #         rewards[no_contact_mask] -= 1.0 * obj_tip_pos_dist_scale
        #     rewards[too_far_mask] -= obj_tip_pos_dist[too_far_mask] * obj_tip_pos_dist_scale
        # else:
        #     print("testing!!! Contact detection is off!!!!!")


        # Zero out rewards for environments that have already succeeded
        # rewards = rewards * (1.0 - successes_env_id_masks)
        self.extras["log"] = {
            "obj_goal_pos_reward": reward_obj_goal_pos.mean(),
            "obj_goal_orn_reward": reward_obj_goal_orn.mean(),
            "tip_obj_orn_reward": reward_tip_obj_orn.mean(),
            # "obj_tip_pos_reward": reward_obj_tip_pos.mean(),
            "successes": successes_env_id_masks.mean(),

        }
        # print("rewards:", rewards)
        return rewards
