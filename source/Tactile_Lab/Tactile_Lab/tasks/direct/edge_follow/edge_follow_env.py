# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
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
# from source.tg3_lab.tg3_lab.tasks.direct import edge
# from isaaclab.utils.math import sample_uniform
from .edge_follow_env_cfg import EdgeFollowEnvCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors.camera import TiledCamera
import cv2
import numpy as np
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, euler_xyz_from_quat, quat_unique
from Tactile_Lab.utility.tactile_sensor import TactileSensor
from Tactile_Lab.utility.utils import positional_encoding, quat_from_rpy, rpy_from_quat
from ipdb import set_trace

class EdgeFollowEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: EdgeFollowEnvCfg

    def __init__(self, cfg: EdgeFollowEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.dist_to_goal = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.dist_from_edge = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.goal_pos_world = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos_scene = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.edge_line_two_end_points_world = torch.zeros((self.num_envs, 2, 3), dtype=torch.float, device=self.device)

        # #################################### For IK ####################################
        self.diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")  # TODO: set the use_relative_mode for RL training
        self.diff_ik_controller = DifferentialIKController(self.diff_ik_cfg, num_envs=self.num_envs, device=self.device)
        # Markers
        if self.cfg.if_debug:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self.ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        # frame_marker_cfg = FRAME_MARKER_CFG.copy()
        # frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self.camera_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/camera_frame"))
        # Define workframe and initial end-effector pose
        self.workframe_scene = torch.tensor(self.cfg.workframe_scene).unsqueeze(0).repeat(self.num_envs, 1).to(self.device)
        self.workframe_quat_scene = quat_from_rpy(self.workframe_scene[:, 3:6])
        self.workframe_scene = torch.cat([self.workframe_scene[:, 0:3], self.workframe_quat_scene], dim=-1)
        self.workframe_world = self.workframe_scene.clone()
        self.workframe_world[:, 0:3] += self.scene.env_origins[:, 0:3]
        self.ee_init_pose_world = self.workframe_scene[0]
        self.ee_init_pose_world = torch.tensor(self.ee_init_pose_world, device=self.device)
        # Track the given command
        # Create buffers to store actions
        self.ik_commands = torch.zeros(self.num_envs, self.diff_ik_controller.action_dim, device=self._robot.device)
        self.ik_commands[:] = self.ee_init_pose_world
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
        self._edge = RigidObject(self.cfg.edge_cfg)
        self._goal = RigidObject(self.cfg.goal_cfg)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["edge"] = self._edge
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
        # set_trace()
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

        actions = F.pad(actions, pad=(0, 4), mode='constant', value=0)  # (N, 6)
        # set_trace()
        if hasattr(self, "actions"):
            self.pre_actions = self.actions.clone()
            if self.pre_actions.shape[1] == 2:
                self.pre_actions = F.pad(self.pre_actions.clone(), pad=(0, 4), mode='constant', value=0)  # (N, 6)
        else:
            self.pre_actions = torch.zeros_like(actions, device=self.device)

        self.actions = actions.clone().clamp(-1.0, 1.0)
        
        self.scaled_actions = (
            self.robot_ee_speed_scale
            * self.dt
            * self.actions
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
        terminated = self.dist_to_goal < self.cfg.success_threshold
        if self.cfg.edge_stop_distance is not None:
            self.terminated_edge_too_far = self.distance_from_edge > self.cfg.edge_stop_distance  # put to cfg
            terminated = terminated | self.terminated_edge_too_far
        else:
            self.terminated_edge_too_far = torch.zeros_like(terminated, dtype=torch.bool)
        # terminated = self._robot.data.joint_pos[:, 3] > 10.39  # TODO: Place holder
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _compute_intermediate_values(self):
        self.ee_pose_world = self._robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]].clone()
        self.ee_pose_scene = self.ee_pose_world[:, 0:3] - self.scene.env_origins
        self.distance_from_edge = self._calculate_distance_from_edge()
        self.dist_to_goal = torch.norm(self.ee_pose_world[:, :2] - self.goal_pos_world[:, :2], dim=-1)

        # compute frame in root frame
        self.ee_pos_work, self.ee_quat_work = subtract_frame_transforms(  # _b: The robot’s base frame
            self.workframe_world[:, 0:3], self.workframe_world[:, 3:7], self.ee_pose_world[:, 0:3], self.ee_pose_world[:, 3:7]
        )
        robot_root_pose_world = self._robot.data.root_pose_w.clone()
        self.ee_pos_b, self.ee_quat_b = subtract_frame_transforms(  # _b: The robot’s base frame, this is equal to ee_pos_scene if robot base frame == scene frame
            robot_root_pose_world[:, 0:3], robot_root_pose_world[:, 3:7], self.ee_pose_world[:, 0:3], self.ee_pose_world[:, 3:7]
        )
        self.ee_vel = self._robot.data.body_lin_vel_w[:, self.robot_entity_cfg.body_ids[0]].clone()

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        # constant_reward = torch.tensor(self.cfg.constant_reward).unsqueeze(0).repeat(self.num_envs).to(self.device)
        return self._compute_rewards(
            # constant_reward,
            self.distance_from_edge,
            self.dist_to_goal,
            self.cfg.edge_dist_reward_scale,
            self.cfg.goal_dist_reward_scale,
            self.cfg.edge_stop_distance,
            self.terminated_edge_too_far,
            self.actions,
            self.cfg.action_penalty_scale,
            self.pre_actions,
            self.cfg.action_rate_penalty_scale,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        ## Reset the robot to the default joint positions, and then reset it to the ee_init_pose_world!
        joint_pos_env_ids = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel_env_ids = self._robot.data.default_joint_vel[env_ids].clone()
        # TODO: Randomize initial joint positions
        # joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
        #     -0.125,
        #     0.125,
        #     (len(env_ids), self._robot.num_joints),
        #     self.device,
        # )
        # joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        self._robot.set_joint_position_target(joint_pos_env_ids, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos_env_ids, joint_vel_env_ids, env_ids=env_ids)
        self._robot.reset(env_ids)  # TODO: Is it necessary to call reset?? Some example in the IsaacLab does not call reset after setting joint states.

        ## Now reset it to the ee_init_pose_world!
        # reset actions
        self.ik_commands[env_ids, :] = self.ee_init_pose_world.clone()
        self.prev_targets[env_ids, :] = self.ee_init_pose_world.clone()
        self.cur_targets[env_ids, :] = self.ee_init_pose_world.clone()
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

        # reset edge object
        self._reset_edge_object(env_ids)
        # reset goal for reward computation
        self._reset_goal(env_ids)
        # compute intermediate values after reset
        self._compute_intermediate_values()

    def _reset_goal(self, env_ids: torch.Tensor | None):
        edge_pos_world_env_ids = self.edge_pos_world.clone()[env_ids]
        edge_ang_env_ids = self.edge_rpy_world[env_ids, 2].clone()
        edge_height = self.cfg.edge_height
        edge_len = self.cfg.edge_length
        goal_z_off_set = 0.2
        # Compute goal position (pose_p) as tensor
        self.goal_pos_world[env_ids, :] = torch.stack([
            edge_pos_world_env_ids[:, 0] + edge_len * torch.cos(edge_ang_env_ids),
            edge_pos_world_env_ids[:, 1] + edge_len * torch.sin(edge_ang_env_ids),
            edge_pos_world_env_ids[:, 2] + edge_height + goal_z_off_set,
        ], dim=-1)  # shape: (N, 3)
        self.goal_pos_scene[env_ids, :] = self.goal_pos_world.clone()[env_ids, :] - self.scene.env_origins[env_ids]
        # Compute two end points of the edge line as tensors
        self.edge_line_two_end_points_world[env_ids, :, :] = torch.stack([
            torch.stack([
                edge_pos_world_env_ids[:, 0] - edge_len * torch.cos(edge_ang_env_ids),
                edge_pos_world_env_ids[:, 1] - edge_len * torch.sin(edge_ang_env_ids),
                edge_pos_world_env_ids[:, 2] + edge_height + goal_z_off_set,
            ], dim=-1),
            self.goal_pos_world[env_ids, :],
        ], dim=1)  # shape: (N, 2, 3)

        goal_root_state = self._goal.data.default_root_state.clone()
        goal_root_state[env_ids, 0:3] = self.goal_pos_world[env_ids, :].clone()
        self._goal.write_root_pose_to_sim(goal_root_state[env_ids, :7], env_ids)
        # self._edge.write_root_velocity_to_sim(goal_root_state[:, 7:])  # No need to set velocity for goal to improve efficiency
        self._goal.reset(env_ids)

        # return goal_pos_world, edge_line_two_end_points_world, edge_ang

    def _reset_edge_object(self, env_ids: torch.Tensor | None):  # TODO: move this to utils file

        edge_root_state_env_ids_scene = self._edge.data.default_root_state.clone()[env_ids]
        edge_root_state_env_ids_world = edge_root_state_env_ids_scene.clone()
        # global object positions (different from isaacgym)
        edge_root_state_env_ids_world[:, 0:3] = (
            edge_root_state_env_ids_scene[:, 0:3] + self.scene.env_origins[env_ids]
        )
        # TODO: If for saving a ref tactile img, move the object away to avoid contact. Need to do it more elegant.
        edge_rpy_env_ids = rpy_from_quat(edge_root_state_env_ids_world[:, 3:7])
        random_angle_scale = 1
        edge_rpy_env_ids[:, 2] = torch.empty_like(edge_rpy_env_ids[:, 2]).uniform_(-random_angle_scale, random_angle_scale) * torch.pi  # 2 * 
        # edge_rpy_env_ids[:, 2] = torch.empty_like(edge_rpy_env_ids[:, 2]).uniform_(-1, 1) * torch.pi  # 2 * 
        edge_root_state_env_ids_world[:, 3:7] = quat_from_rpy(edge_rpy_env_ids)

        self._edge.write_root_pose_to_sim(edge_root_state_env_ids_world[:, :7], env_ids)
        # self._edge.write_root_velocity_to_sim(edge_root_state_env_ids_world[:, 7:])  # No need to set velocity for goal to improve efficiency
        self._edge.reset(env_ids)
        self.edge_rpy_world = rpy_from_quat(self._edge.data.root_quat_w.clone())
        self.edge_pos_world = self._edge.data.root_pos_w.clone()

    def _calculate_distance_from_edge(self) -> torch.Tensor:  # TODO: move this to utils file (or not?)
        # use only x/y dont need z
        p1 = self.edge_line_two_end_points_world[:, 0, :2]  # shape: (N, 2)
        p2 = self.edge_line_two_end_points_world[:, 1, :2]  # shape: (N, 2)
        p3 = self.ee_pose_world[:, :2]                # shape: (N, 2)
        # vector from p1 to p2
        v1 = p2 - p1  # shape: (N, 2)
        # vector from p1 to p3
        v2 = p1 - p3  # shape: (N, 2), broadcasted

        # 2D cross product: z = x1*y2 - y1*x2
        cross_z = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]  # shape: (N,)

        # Norm of v1
        norm_v1 = torch.norm(v1, dim=1) + 1e-8  # shape: (N,), avoid divide-by-zero

        # Perpendicular distance from p3 to each edge line
        dist = torch.abs(cross_z) / norm_v1  # shape: (N,)
        return dist.to(self.device)
    
    def _get_oracle_obs(self) -> dict[str, torch.Tensor]:
        """
        Get the oracle observations. All in world frame. #TODO: Change to workframe when deploy to real robot
        """
        if self.cfg.if_use_positional_encoding:
            enc_tcp_xy = positional_encoding(self.ee_pos_work[:, :2].clone())
            enc_tcp_vel_xy = positional_encoding(self.ee_vel[:, :2].clone())
            enc_goal_xy = positional_encoding(self.goal_pos_world[:, :2].clone())
            enc_edge_angle = positional_encoding(self.edge_rpy_world[:, 2].unsqueeze(1).clone())
        # Concatenate to form observation buffer
            return torch.cat((enc_tcp_xy, enc_tcp_vel_xy, enc_goal_xy, enc_edge_angle), dim=-1)  # Dimension: 2*8 + 2*8 + 2*8 + 8 = 56
        else:
            # set_trace()
            # instead of using goal_pos_world, use goal_pos_scene, otherwise the obs info will be lost by clamping in large env numbers
            return torch.cat((self.ee_pos_work[:, :2], self.ee_vel[:, :2], self.goal_pos_scene[:, :2], self.edge_rpy_world[:, 2].unsqueeze(1)), dim=-1)  # Dimension: 2 + 2 + 2 + 1 = 7

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
            dummy_1d = torch.zeros((img.shape[0], 1), device=img.device)

            return {
                "image": img,          # CNN input (B, C, H, W)
                "dummy_1d": dummy_1d   # MLP input (B, 1)
            }

    def _compute_rewards(
        self,
        # constant_reward: torch.Tensor,
        distance_from_edge: torch.Tensor,
        dist_to_goal: torch.Tensor,
        edge_dist_reward_scale: float,
        goal_dist_reward_scale: float,
        edge_stop_distance: float | None,
        terminated_edge_too_far: torch.Tensor,
        actions: torch.Tensor,
        action_penalty_scale: float,
        pre_actions: torch.Tensor,
        action_rate_penalty_scale: float,

    ):
        terminal_penalty = -300.0
        action_penalty = torch.sum(torch.square(actions), dim=1)
        action_rate_l2 = torch.sum(torch.square(actions - pre_actions), dim=1)

        rewards = -(
            edge_dist_reward_scale * distance_from_edge
            + goal_dist_reward_scale * dist_to_goal
            + action_penalty_scale * action_penalty
            + action_rate_penalty_scale * action_rate_l2
        )
        # add terminal penalty
        if edge_stop_distance is not None:
            rewards[terminated_edge_too_far] += terminal_penalty

        # Logging
        self.extras["log"] = {
            "scaled_dist_edge_reward": (edge_dist_reward_scale * distance_from_edge).mean(),
            "scaled_dist_goal_reward": (goal_dist_reward_scale * dist_to_goal).mean(),
            "action_penalty": (action_penalty_scale * action_penalty).mean(),
            "action_rate_penalty": (action_rate_penalty_scale * action_rate_l2).mean(),
            "terminated_edge_too_far_rate": terminated_edge_too_far.float().mean(),
        }

        # Terminal penalty logging
        if edge_stop_distance is None:
            self.extras["log"]["terminal_penalty_enabled"] = 0.0
            self.extras["log"]["terminal_penalty_mean"] = 0.0
        else:
            self.extras["log"]["terminal_penalty_enabled"] = 1.0
            self.extras["log"]["terminal_penalty_mean"] = (
                terminal_penalty * terminated_edge_too_far.float()
            ).mean()

        return rewards