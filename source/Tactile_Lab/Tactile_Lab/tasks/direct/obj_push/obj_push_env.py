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
from .obj_push_env_cfg import ObjPushEnvCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors.camera import TiledCamera
from isaaclab.sensors import ContactSensor
from Tactile_Lab.utility.tactile_sensor import TactileSensor
from Tactile_Lab.utility.utils import quat_from_rpy, rpy_from_quat

import cv2
import numpy as np
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, euler_xyz_from_quat, quat_unique, quat_apply
from isaaclab.utils.math import sample_uniform

class ObjPushEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: ObjPushEnvCfg

    def __init__(self, cfg: ObjPushEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        # Initialize goal-related variables
        self.current_goal_list_id_all_envs = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )        
        self.current_goal_pose_worldframe_all_envs = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)) # (N, 6)
        self.current_goal_quat_worldframe_all_envs = torch.tensor([0, 0, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)) # (N, 4)
        self.goals_pose_worldframe_all_envs = torch.zeros(
            (self.num_envs, self.cfg.traj_n_points, 7), dtype=torch.float, device=self.device
        )
        self.successes_env_id_masks = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.int64)

        # Markers
        if self.cfg.if_debug:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            # self.ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
            # frame_marker_cfg = FRAME_MARKER_CFG.copy()
            # frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            # self.camera_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/camera_frame"))
            self.obj_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/obj_frame"))
            self.goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/goal_frame"))

        # #################################### For IK ####################################
        self.diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")  # TODO: set the use_relative_mode for RL training
        self.diff_ik_controller = DifferentialIKController(self.diff_ik_cfg, num_envs=self.num_envs, device=self.device)

        # Define workframe, robot_base_frame, and scene_frame, and initial end-effector pose
        self.workframe_in_base_frame = torch.tensor(self.cfg.workframe_in_base_frame).unsqueeze(0).repeat(self.num_envs, 1).to(self.device)
        self.robot_base_in_scene_frame = torch.tensor(self.cfg.robot_base_in_scene_frame).unsqueeze(0).repeat(self.num_envs, 1).to(self.device)

        # self.workframe_quat_scene = quat_from_rpy(self.workframe_in_base_frame[:, 3:6])
        # self.workframe_in_base_frame = torch.cat([self.workframe_in_base_frame[:, 0:3], self.workframe_quat_scene], dim=-1)
        self.workframe_in_scene_frame = self.workframe_in_base_frame.clone()

        self.workframe_in_scene_frame_pos, self.workframe_in_scene_frame_quat = combine_frame_transforms(  # _b: The robot’s base frame, this is equal to ee_pos_scene if robot base frame == scene frame
            self.robot_base_in_scene_frame[:, 0:3], self.robot_base_in_scene_frame[:, 3:7],
            self.workframe_in_base_frame[:, 0:3], self.workframe_in_base_frame[:, 3:7]
        )
        self.workframe_in_scene_frame = torch.cat([self.workframe_in_scene_frame_pos[:, 0:3], self.workframe_in_scene_frame_quat], dim=-1)
        self.scene_quats = torch.zeros((self.num_envs, 4), device=self.device)
        self.scene_quats[:, 0] = 1.0   # (w, x, y, z), assume the scene frame orientation is aligned with the world frame
        self.workframe_world = self.workframe_in_scene_frame.clone()
        self.workframe_world[:, 0:3] += self.scene.env_origins[:, 0:3] # assume the scene frame orientation is aligned with the world frame
        self.ee_init_pose_base = self.workframe_in_base_frame[0]
        self.ee_init_pose_base = torch.tensor(self.ee_init_pose_base, device=self.device)
        # Track the given command
        # Create buffers to store actions
        self.ik_commands = torch.zeros(self.num_envs, self.diff_ik_controller.action_dim, device=self._robot.device)
        self.ik_commands[:] = self.ee_init_pose_base
        if self.cfg.robot_name == "franka_panda":
            self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        elif self.cfg.robot_name == "ur10":
            self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
        elif self.cfg.robot_name == "ur5_tactip":
            self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["tcp_link"])
        self.robot_entity_cfg.resolve(self.scene)
        # Obtain the frame index of the end-effector
        # For a fixed base robot, the frame index is one less than the body index. This is because
        # the root body is not included in the returned Jacobians.
        if self._robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

    def _setup_scene(self):
        self.obs_type = self.cfg.obs_type
        self._robot = Articulation(self.cfg.robot_cfg)
        self._object = RigidObject(self.cfg.object_cfg)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        # self.scene.rigid_objects["table"] = self._table
        self.scene.rigid_objects["object"] = self._object
        if self.obs_type == "tactile":
            self.tactile_sensor_cfg = self.cfg.tactile_sensor_cfg
            # self._tactile_camera = Camera(self.cfg.tactile_camera)
            self._tactile_camera = TiledCamera(self.tactile_sensor_cfg.tactile_camera)
            self.scene.sensors["tactile_camera"] = self._tactile_camera
            # Flags for initialize the tactile camera reference images
            self._if_loaded_tactile_reference_image = False
            self.tactile_sensor = TactileSensor(self._tactile_camera, self.tactile_sensor_cfg, self.device, self.num_envs, __file__)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.tcp_lims = torch.tensor(self.cfg.tcp_lims_single).unsqueeze(0).repeat(self.num_envs, 1, 1).to(self.device)
        self.robot_ee_speed_scale = torch.tensor(self.cfg.robot_ee_speed_scale).to(self.device)

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
        self.goals_pose_worldframe_all_envs[env_ids] = goals_pose_in_worldframe[env_ids]
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
        traj_pos_workframe_all_envs = torch.zeros((self.num_envs, self.cfg.traj_n_points, 3), dtype=torch.float, device=self.device)
        traj_rpy_workframe_all_envs = torch.zeros((self.num_envs, self.cfg.traj_n_points, 3), dtype=torch.float, device=self.device)
        n_pts = self.cfg.traj_n_points
        device = self.device

        # (num_envs,) random angle per env
        traj_ang = torch.rand(self.num_envs, device=device) * (torch.pi / 4) - (torch.pi / 8)
        # scalar
        init_offset = self.cfg.object_size / 2 + self.cfg.traj_spacing

        # (traj_n_points,)
        dist = torch.arange(n_pts, device=device, dtype=torch.float) * self.cfg.traj_spacing

        # directions: (num_envs,)
        dir_y = torch.cos(traj_ang)
        dir_x = -torch.sin(traj_ang)

        # broadcast to (num_envs, traj_n_points)
        x = dist[None, :] * dir_x[:, None]
        y = init_offset + dist[None, :] * dir_y[:, None]
        z = torch.full((self.num_envs, n_pts), self.cfg.goal_z_pos, device=device)

        # stack → (num_envs, traj_n_points, 3)
        traj_pos_workframe_all_envs[:] = torch.stack([x, y, z], dim=-1)
        # calc orientation to place object at
        x = traj_pos_workframe_all_envs[:, :, 0]   # [num_envs, T]
        y = traj_pos_workframe_all_envs[:, :, 1]   # [num_envs, T]

        # gradient along trajectory axis (dim=1)
        dx = torch.gradient(x, dim=1)[0]  # [num_envs, T]
        dy = torch.gradient(y, dim=1)[0]  # [num_envs, T]

        traj_rpy_workframe_all_envs[:, :, 2] = (
            torch.atan2(dy, dx)
            - torch.pi / 2
        )

        return traj_pos_workframe_all_envs, traj_rpy_workframe_all_envs
    
    def check_TCP_vel_lims(self, vels):
        """
        check whether action will take TCP outside of limits,
        zero any velocities that will.
        """
        # print("vels before capping:", vels)
        self.ee_rpy_work = rpy_from_quat(self.ee_quat_work)
        ee_pose_work = torch.cat((self.ee_pos_work, self.ee_rpy_work), dim=-1)  # (N, 6)
        # get bool arrays for if limits are exceeded and if velocity is in
        # the direction that's exceeded
        # Unpack limits: shape → (num_envs, 6)
        # Unpack limits: (num_envs, 6)
        llims = self.tcp_lims[:, :, 0]
        ulims = self.tcp_lims[:, :, 1]
        # Identify violations based on current position and velocity direction
        exceed_llims = torch.logical_and(ee_pose_work < llims, vels < 0)
        exceed_ulims = torch.logical_and(ee_pose_work > ulims, vels > 0)
        exceeded = torch.logical_or(exceed_llims, exceed_ulims)  # shape: (num_envs, 6)
        # Cap velocities: set to 0 where limit is exceeded
        capped_vels = vels.clone()
        capped_vels[exceeded] = 0.0
        # print("capped_vels:", capped_vels)
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

    def transform_action_to_tcp_frame(self, actions: torch.Tensor):
        perp_vector = torch.tensor([1., 0., 0.], device=self.device).expand(self.num_envs, -1)   # "perpendicular"
        par_vector = torch.tensor([0., 1., 0.], device=self.device).expand(self.num_envs, -1)   # "parallel" (outwards)
        # rotate local vectors into world frame (expects self.tip_orn: [B, 4], returns [B, 3])
        perp_tip_dir_world = quat_apply(self.ee_pose_scene[:, 3:7], perp_vector)   # [B, 3]
        par_tip_dir_world = quat_apply(self.ee_pose_scene[:, 3:7], par_vector)    # [B, 3]
        # scales from actions (shape: [B])
        perp_scale = actions[:, 0]                                    # [B]
        par_scale = actions[:, 1]                                    # [B]
        # scale direction vectors (broadcast to [B, 3])
        perp_action_world = perp_tip_dir_world * perp_scale.unsqueeze(-1)
        par_action_world = par_tip_dir_world * par_scale.unsqueeze(-1)
        # accumulate into the outgoing action vector
        total_xy = perp_action_world[:, :2] + par_action_world[:, :2]  # [B, 2]
        actions[:, 0] = total_xy[:, 0]
        actions[:, 1] = total_xy[:, 1]
        return actions

    def _pre_physics_step(self, actions: torch.Tensor):

        # obtain quantities from simulation
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        # self.ee_pose_world = self._robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]].clone()  # no need, already obtained in _compute_intermediate_values
        robot_root_pose_world = self._robot.data.root_pose_w.clone()
        
        joint_pos = self._robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids].clone()
        # compute frame in root frame
        self.ee_pos_work, self.ee_quat_work = subtract_frame_transforms(  # _b: The robot’s base frame
            self.workframe_world[:, 0:3], self.workframe_world[:, 3:7], self.ee_pose_world[:, 0:3], self.ee_pose_world[:, 3:7]
        )
        self.ee_pos_b, self.ee_quat_b = subtract_frame_transforms(  # _b: The robot’s base frame, this is equal to ee_pos_scene if robot base frame == scene frame
            robot_root_pose_world[:, 0:3], robot_root_pose_world[:, 3:7], self.ee_pose_world[:, 0:3], self.ee_pose_world[:, 3:7]
        )

        # TODO: Test action hardcoding for debugging
        # print("testing action hardcoding for debugging in PushEnv!!")
        # actions = torch.tensor([[0, 1]], device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)
        actions = F.pad(actions, pad=(0, 4), mode='constant', value=0)  # (N, 6)
        actions[:, [1, 5]] = actions[:, [5, 1]]  # TODO: this is a hack to swap the z and pitch axes, need to fix this properly

        if hasattr(self, "actions"):
            self.pre_actions = self.actions.clone()
            if self.pre_actions.shape[1] == 2:
                self.pre_actions = F.pad(self.pre_actions.clone(), pad=(0, 4), mode='constant', value=0)  # (N, 6)
        else:
            self.pre_actions = torch.zeros_like(actions, device=self.device)
        actions[:, 1] = 1.0  # push obj along y axis
        # actions[0, 5] *= 5.0  # TODO: Have a new scale for the pitch axis, this is a hack to make it work for now
        self.actions = actions.clone().clamp(-self.cfg.clip_actions, self.cfg.clip_actions)
        self.tcp_actions = self.transform_action_to_tcp_frame(self.actions)
        self.scaled_actions = (
            self.robot_ee_speed_scale
            * self.dt
            * self.tcp_actions
            * self.cfg.action_scale
        )
        self.checked_scaled_actions = self.check_TCP_vel_lims(self.scaled_actions)
        self.cur_targets = self.compute_target_pose(self.prev_targets, self.checked_scaled_actions)
        self.prev_targets[:] = self.cur_targets.clone()
        self.ik_commands = self.cur_targets
        self.diff_ik_controller.set_command(self.ik_commands)

        self.robot_dof_targets = self.diff_ik_controller.compute(self.ee_pos_b, self.ee_quat_b, jacobian, joint_pos)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets, joint_ids=self.robot_entity_cfg.joint_ids)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()  # TODO: check whether to put this in get_done or get_reward!!!
        terminated = self.successes_env_id_masks.bool() | self.too_far_mask
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    def cos_tcp_dist_to_obj(self):
        """
        Calculate the cosine distance between the TCP and the object.
        This is calculated as the dot product of the TCP orientation and the object orientation.
        """
        cur_obj_orn_worldframe = self._object.data.root_quat_w
        cur_tcp_orn_worldframe = self.ee_pose_world[:, 3:7]

        # get normal vector of object
        q_obj = cur_obj_orn_worldframe
        init_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device).repeat(q_obj.shape[0], 1)
        obj_vector = quat_apply(q_obj, init_vec)   # [B,3]

        # get vector of t_s tip, directed through tip body
        q_tip = cur_tcp_orn_worldframe
        tip_init_vector = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device).repeat(q_tip.shape[0], 1)
        tip_vector = quat_apply(q_tip, tip_init_vector)   # [B,3]
        # get the cosine similarity/distance between the two vectors
        # Dot product along the last dimension
        cos_sim = (obj_vector * tip_vector).sum(dim=-1) / (
            obj_vector.norm(dim=-1) * tip_vector.norm(dim=-1)
        )
        # Cosine distance
        cos_dist = 1 - cos_sim
        # print("cos_dist:", cos_dist)
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
        self.ee_pose_world = self._robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]].clone()
        self.ee_pose_scene = self.ee_pose_world[:, 0:3] - self.scene.env_origins
        ee_pos_scene, ee_quat_scene = subtract_frame_transforms(
            self.scene.env_origins,
            self.scene_quats,
            self.ee_pose_world[:, 0:3],
            self.ee_pose_world[:, 3:7],
        )
        self.ee_pose_scene = torch.cat([ee_pos_scene, ee_quat_scene], dim=-1)
        # ee coordinates in workframe
        self.ee_pos_work, self.ee_quat_work = subtract_frame_transforms(  # _b: The robot’s base frame
            self.workframe_world[:, 0:3], self.workframe_world[:, 3:7], self.ee_pose_world[:, 0:3], self.ee_pose_world[:, 3:7]
        )
        self.ee_rpy_work = rpy_from_quat(self.ee_quat_work)
        # ee coordinates in base frame
        self.ee_linvel = self._robot.data.body_lin_vel_w[:, self.robot_entity_cfg.body_ids[0]].clone()
        self.ee_angvel = self._robot.data.body_ang_vel_w[:, self.robot_entity_cfg.body_ids[0]].clone()
        # data for object
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
        self.obj_tip_pos_dist = self.xy_obj_dist_to_tip()
        self.obj_goal_pos_dist = self.xy_obj_dist_to_goal()
        self.obj_goal_orn_dist = self.orn_obj_dist_to_goal()
        self.tip_obj_orn_dist = self.cos_tcp_dist_to_obj()
        ###################################### data for goals ####################################
        self.goal_update(self.get_sub_goal_update_env_ids())
        ####################################contact sensor####################################
        self.tip_contact_force = self._contact_sensor.data.net_forces_w
        tip_force_mag = self.tip_contact_force.norm(dim=-1)               # [B]
        self.no_contact_tip_mask = tip_force_mag <= 1e-6              # near-zero force
        self.too_far_mask = (self.obj_tip_pos_dist > self.cfg.terminatie_dist_obj_tip) & self.no_contact_tip_mask.view(-1)

        if self.cfg.if_debug:
            self.obj_marker.visualize(self._object.data.root_pos_w, self._object.data.root_quat_w)
            self.goal_marker.visualize(
                self.current_goal_pose_worldframe_all_envs[:, 0:3],
                self.current_goal_quat_worldframe_all_envs,
            )


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
        # torch.cuda.synchronize()
        # t1 = time.perf_counter()
        # print(f"period 1 cost = {(t1 - t0)*1e3:.3f} ms")
        
        active_subgoal_ids = self.current_goal_list_id_all_envs[active_env_ids]
        achieved_subgoal_ids = updated_ids[active_mask] - 1

        for subgoal_id in range(self.cfg.traj_n_points):
            # ---- current sub-goal (buffers) ----
            mask_current = active_subgoal_ids == subgoal_id
            env_ids_current = active_env_ids[mask_current]

            if env_ids_current.numel() > 0:
                goal_pose = self.goals_pose_worldframe_all_envs[env_ids_current, subgoal_id]
                pos = goal_pose[:, 0:3]
                quat = goal_pose[:, 3:7]
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
            self.obj_tip_pos_dist,
            self.cfg.obj_goal_pos_dist_scale,
            self.cfg.obj_goal_orn_dist_scale,
            self.cfg.tip_obj_orn_dist_scale,
            self.cfg.obj_tip_pos_dist_scale,
            self.successes_env_id_masks,
            self.too_far_mask,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        ################################ Reset robot ###############################################
        ## Reset the robot to the default joint positions, and then reset it to the ee_init_pose_base!
        joint_pos_env_ids = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel_env_ids = self._robot.data.default_joint_vel[env_ids].clone()
        self._robot.set_joint_position_target(joint_pos_env_ids, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos_env_ids, joint_vel_env_ids, env_ids=env_ids)
        self._robot.reset(env_ids)  # TODO: Is it necessary to call reset?? Some example in the IsaacLab does not call reset after setting joint states.

        ## Now reset it to the ee_init_pose_base! Question: is the ik_commands refer to world frame or scene frame or robot base frame? It is the base frame!
        ############################# reset actions and ik commands #############################
        self.ik_commands[env_ids, :] = self.ee_init_pose_base.clone()
        self.prev_targets[env_ids, :] = self.ee_init_pose_base.clone()
        self.cur_targets[env_ids, :] = self.ee_init_pose_base.clone()
        # reset controller
        self.diff_ik_controller.reset(env_ids)
        self.diff_ik_controller.set_command(self.ik_commands)
        # obtain quantities from simulation
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        self.ee_pose_world = self._robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]].clone()
        self.ee_pose_scene = self.ee_pose_world[:, 0:3] - self.scene.env_origins
        robot_root_pose_world = self._robot.data.root_pose_w.clone()
        joint_pos = self._robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids].clone()
        # compute frame in root frame
        self.ee_pos_b, self.ee_quat_b = subtract_frame_transforms(  # _b: The robot’s base frame. This already convert the ee pose to local scene/robot frame
            robot_root_pose_world[:, 0:3], robot_root_pose_world[:, 3:7], self.ee_pose_world[:, 0:3], self.ee_pose_world[:, 3:7]
        )
        
        self.robot_dof_targets = self.diff_ik_controller.compute(self.ee_pos_b, self.ee_quat_b, jacobian, joint_pos)
        joint_vel = self._robot.data.default_joint_vel
        self._robot.set_joint_position_target(self.robot_dof_targets[env_ids, :], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(self.robot_dof_targets[env_ids, :], joint_vel[env_ids, :], env_ids=env_ids)
        self._robot.reset(env_ids)  # TODO: Is it necessary to call reset?? Some example in the IsaacLab does not call reset after setting joint states.

        ####################### reset object#########################
        obj_default_state_env_ids_scene = self._object.data.default_root_state.clone()[env_ids]
        # Convert state in scene frame to world frame
        obj_default_state_env_ids_world = obj_default_state_env_ids_scene.clone()
        obj_default_state_env_ids_world[:, 0:3] = (
            obj_default_state_env_ids_scene[:, 0:3] + self.scene.env_origins[env_ids]
        )
        obj_default_state_env_ids_world[:, 7:] = torch.zeros_like(self._object.data.default_root_state[env_ids, 7:]) # Set all the velocities to zero
        self._object.write_root_pose_to_sim(obj_default_state_env_ids_world[:, :7], env_ids)
        self._object.write_root_velocity_to_sim(obj_default_state_env_ids_world[:, 7:], env_ids)

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
        oracle_obs = torch.cat((self.ee_pos_work[:, :2],
                                self.ee_rpy_work[:, 2].unsqueeze(-1),
                                self.ee_linvel[:, :2],
                                self.ee_angvel[:, 2].unsqueeze(-1),
                                current_goal_pos_xy_work,
                                goal_yaw.unsqueeze(-1),
                                obj_pos_xy_work,
                                obj_yaw.unsqueeze(-1),
                                self.obj_linvel[:, :2],
                                self.obj_angvel[:, 2].unsqueeze(-1)
                                ), dim=-1)
        # print("current_goal_pos_xy_work:", current_goal_pos_xy_work)
        return oracle_obs

    def get_proprioception_and_goal_for_obs_dict(self) -> torch.Tensor:
        proprioception_and_goal = torch.cat((
            self.ee_pos_work[:, :2],
            self.ee_rpy_work[:, 2].unsqueeze(-1),
            self.current_goal_pos_work[:, :2],
            self.current_goal_rpy_work[:, 2].unsqueeze(-1)
        ), dim=-1)
        return proprioception_and_goal

    def _get_observations(self) -> dict:
        if self.obs_type == "oracle":
            obs = self._get_oracle_obs()
            return {"oracle": torch.clamp(obs, -5.0, 5.0)}  # TODO: Finetune the clamp value (for oracle, max=1, min=-1 due to positional encoding. For tactile image? ). This clamping is for numerical stability
        elif self.obs_type == "tactile":
            if self.tactile_sensor_cfg.if_save_tactile_reference_image:
                self.tactile_sensor._save_tactile_reference_images()
            elif not self._if_loaded_tactile_reference_image:  # load only once
                self._if_loaded_tactile_reference_image = True
                self.tactile_sensor._load_tactile_reference_images()
            tactile_obs = self.tactile_sensor._get_tactile_images_tensors()
            tactile_obs = tactile_obs.permute(0, 3, 1, 2)  # [B, H, W, C] → [B, C, H, W], for rsl-rl
            img = torch.clamp(tactile_obs, -1.0, 1.0)

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
        obj_tip_pos_dist: torch.Tensor,
        obj_goal_pos_dist_scale: float,
        obj_goal_orn_dist_scale: float,
        tip_obj_orn_dist_scale: float,
        obj_tip_pos_dist_scale: float,
        successes_env_id_masks: torch.Tensor,
        too_far_mask: torch.Tensor,
    ):
        contact_detection_off = False

        # Reward for reaching the goal position
        reward_obj_goal_pos = -obj_goal_pos_dist_scale * obj_goal_pos_dist
        # Reward for reaching the goal orientation
        reward_obj_goal_orn = -obj_goal_orn_dist_scale * obj_goal_orn_dist
        # Reward for aligning the tip orientation with the object orientation
        reward_tip_obj_orn = -tip_obj_orn_dist_scale * tip_obj_orn_dist
        # Reward for bringing the tip close to the object
        reward_obj_tip_pos = -obj_tip_pos_dist_scale * obj_tip_pos_dist

        rewards = (
            reward_obj_goal_pos
            + reward_obj_goal_orn
            + reward_tip_obj_orn
            # + reward_obj_tip_pos
        )

        if not contact_detection_off:
            rewards[too_far_mask] -= obj_tip_pos_dist[too_far_mask] * obj_tip_pos_dist_scale
        else:
            print("testing!!! Contact detection is off!!!!!")


        # Zero out rewards for environments that have already succeeded
        # rewards = rewards * (1.0 - successes_env_id_masks)
        self.extras["log"] = {
            "obj_goal_pos_reward": reward_obj_goal_pos.mean(),
            "obj_goal_orn_reward": reward_obj_goal_orn.mean(),
            "tip_obj_orn_reward": reward_tip_obj_orn.mean(),
            "obj_tip_pos_reward": reward_obj_tip_pos.mean(),
            "successes": successes_env_id_masks.mean(),

        }
        # print("rewards:", rewards)
        return rewards
