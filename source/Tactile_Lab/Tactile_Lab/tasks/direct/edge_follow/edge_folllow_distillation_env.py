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
from .edge_distillation_env_cfg import EdgeDistillationEnvCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors.camera import TiledCamera
import cv2
import numpy as np
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, euler_xyz_from_quat, quat_unique

class EdgeDistillationEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: EdgeDistillationEnvCfg

    def __init__(self, cfg: EdgeDistillationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
        # Create auxiliary variables for computing applied action, observations and rewards
        # self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        # self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        # self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        # self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        # self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1
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
        self.workframe_quat_scene = self._quat_from_rpy(self.workframe_scene[:, 3:6])
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
            # self._tactile_camera = Camera(self.cfg.tactile_camera)
            self._tactile_camera = TiledCamera(self.cfg.tactile_camera)
            self.scene.sensors["tactile_camera"] = self._tactile_camera
            # Flags for initialize the tactile camera reference images
            self._if_loaded_tactile_reference_image = False
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
        self.ee_rpy_work = self._rpy_from_quat(self.ee_quat_work)
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
        pre_targets_rpy = self._rpy_from_quat(pre_targets_pose[:, 3:7])
        targets_rpy = pre_targets_rpy + actions[:, 3:6]
        target_quat = self._quat_from_rpy(targets_rpy)
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

        # TODO: Test action hardcoding for debugging
        # actions = torch.tensor([[-1, 0, 0, 0, 0, 0]], device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)
        actions = F.pad(actions, pad=(0, 4), mode='constant', value=0)  # (N, 6)
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
        # targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        # self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        # # compute the joint commands
        # actions
        # ep = int(self.episode_length_buf[0])
        # ep = self.episode_length_buf[0]

        # # +X movement: 300–600
        # if ep < 600:
        #     self.ik_commands[0][0] += 0.0002
        # # stop: 600–700

        # elif ep < 1800:
        #     euler_angle = euler_xyz_from_quat(self.ik_commands[:, 3:7])
        #     euler_angle_tensor = torch.stack(euler_angle, dim=-1)
        #     euler_angle_tensor[0][2] -= 0.01
        #     roll, pitch, yaw = euler_angle_tensor.unbind(dim=-1)
        #     euler_angle = (roll, pitch, yaw)
        #     self.ik_commands[0][3:7] = quat_from_euler_xyz(*euler_angle)[0]
        self.ik_commands = self.cur_targets
        self.diff_ik_controller.set_command(self.ik_commands)

        self.robot_dof_targets = self.diff_ik_controller.compute(self.ee_pos_b, self.ee_quat_b, jacobian, joint_pos)

    # def _convert_to_direct_action(self, command):
    #     # Convert teleop command to joint-space actions
    #     joint_positions = self.ik_solver.solve(command)
    #     return joint_positions  # Direct env expects raw joint commands

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
        # print("goal_pos_scene:", self.goal_pos_scene[env_ids, :])
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
        edge_rpy_env_ids = self._rpy_from_quat(edge_root_state_env_ids_world[:, 3:7])

        random_angle_scale = 1
        edge_rpy_env_ids[:, 2] = torch.empty_like(edge_rpy_env_ids[:, 2]).uniform_(-random_angle_scale, random_angle_scale) * torch.pi  # 2 * 
        # edge_rpy_env_ids[:, 2] = torch.empty_like(edge_rpy_env_ids[:, 2]).uniform_(-1, 1) * torch.pi  # 2 * 
        edge_root_state_env_ids_world[:, 3:7] = self._quat_from_rpy(edge_rpy_env_ids)

        self._edge.write_root_pose_to_sim(edge_root_state_env_ids_world[:, :7], env_ids)
        # self._edge.write_root_velocity_to_sim(edge_root_state_env_ids_world[:, 7:])  # No need to set velocity for goal to improve efficiency
        self._edge.reset(env_ids)
        self.edge_rpy_world = self._rpy_from_quat(self._edge.data.root_quat_w.clone())
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
    
    def _plot_tactile_depth_image(self, tactile_depth_image_tensor, pause_time=1):
        """
        Plot the tactile depth image.

        Args:
            tactile_depth_image_tensor (torch.Tensor): Tactile depth image tensor.
        """
        # depth_vis = (tactile_depth_image_tensor - tactile_depth_image_tensor.min()) / (tactile_depth_image_tensor.max() - tactile_depth_image_tensor.min() + 1e-6)
        import matplotlib.pyplot as plt
        if isinstance(tactile_depth_image_tensor, np.ndarray):
            plt.imshow(tactile_depth_image_tensor, cmap='gray'); plt.title("Tactile Depth Image"); plt.pause(pause_time)
        else:
            plt.imshow(tactile_depth_image_tensor.cpu().numpy(), cmap='gray'); plt.title("Tactile Depth Image"); plt.pause(pause_time)
            plt.close()

    def _get_tactile_reference_img_path(self):
        saved_file_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../../../reference_images',
            self.cfg.sensor_type,
            str(self.cfg.tactile_img_size) + "x" + str(self.cfg.tactile_img_size),
        )
        return saved_file_dir

    def _load_tactile_reference_images(self):
        # get saved reference images
        saved_file_dir = self._get_tactile_reference_img_path()
        nodef_gray_savefile = os.path.join(saved_file_dir, "nodef_dep.npy")
        # load border images from simulation
        self.no_deformation_image = np.load(nodef_gray_savefile)
        self.no_deformation_image = torch.from_numpy(self.no_deformation_image).to(self.device)

        # For debugging, visualize the no deformation image loaded from file
        # self._plot_tactile_depth_image(self.no_deformation_image, 0.01)

    def _save_tactile_reference_images(self):
        # grab images for creating border from simulation
        if self.num_envs != 1:
            raise ValueError("Reference images can only be saved for a single environment.")

        no_deformation_image = self._tactile_camera.data.output["distance_to_image_plane"][0].cpu().numpy()
        saved_file_dir = self._get_tactile_reference_img_path()
        # # create new directory
        os.makedirs(saved_file_dir, exist_ok=True)
        # # save file names
        nodef_dep_savefile = os.path.join(saved_file_dir, "nodef_dep.npy")
        # save border images from simulation
        np.save(nodef_dep_savefile, no_deformation_image)
        # np.save(nodef_dep_savefile, no_deformation_dep)
        # For debugging, visualize the no deformation image
        self._plot_tactile_depth_image(no_deformation_image, 0.01)
        print(f"Reference images saved to {saved_file_dir}")
        print("Safely exit the script and remember to comment out this function 'save_reference_images'.")
        exit()

    def _get_oracle_obs(self) -> dict[str, torch.Tensor]:
        """
        Get the oracle observations. All in world frame. #TODO: Change to workframe when deploy to real robot
        """
        # goal_x_y = self.goal_pos_tensors[:, :2]  # (N, 2)
        # Encode each part
        # print("ee_pos_work:", self.ee_pos_work[:, :2])
        # print("goal_pos_world:", self.goal_pos_world[:, :2])
        # print("edge angle:", self.edge_rpy_world[:, 2])
        if self.cfg.if_use_positional_encoding:
            enc_tcp_xy = self._positional_encoding(self.ee_pos_work[:, :2].clone())
            enc_tcp_vel_xy = self._positional_encoding(self.ee_vel[:, :2].clone())
            enc_goal_xy = self._positional_encoding(self.goal_pos_world[:, :2].clone())
            enc_edge_angle = self._positional_encoding(self.edge_rpy_world[:, 2].unsqueeze(1).clone())
        # Concatenate to form observation buffer
            return torch.cat((enc_tcp_xy, enc_tcp_vel_xy, enc_goal_xy, enc_edge_angle), dim=-1)  # Dimension: 2*8 + 2*8 + 2*8 + 8 = 56
        else:
            # instead of using goal_pos_world, use goal_pos_scene, otherwise the obs info will be lost by clamping in large env numbers
            return torch.cat((self.ee_pos_work[:, :2], self.ee_vel[:, :2], self.goal_pos_scene[:, :2], self.edge_rpy_world[:, 2].unsqueeze(1)), dim=-1)  # Dimension: 2 + 2 + 2 + 1 = 7

    def _get_tactile_images_tensors(self):  # TODO: Optimize this function to avoid redundant computation for training
        """
        Returns processed tactile image tensor (float32, [0,1]).
        """
        if self.cfg.tactile_image_type == 'depth_original':
            cam_img = self._tactile_camera.data.output["distance_to_image_plane"]
            if cam_img.max() > 20:  # to avoid invalid depth values
                img = torch.clamp(cam_img, 0, 1.0)
            cam_min, cam_max = cam_img.min(), cam_img.max()
            img = ((cam_img - cam_min) / (cam_max - cam_min) * 255).to(torch.uint8)
        else:
            if 'depth' in self.cfg.tactile_image_type:
                cam_img_original = self._tactile_camera.data.output["distance_to_image_plane"]
                cam_img = torch.clamp(torch.abs(cam_img_original - self.no_deformation_image) * self.cfg.tactile_depth_enhance_scale , 0, 1.0)  # scale the depth difference to [0, 1] (also clamp it if exceeds 1.0)
                # cam_img = torch.abs(cam_img_original - self.no_deformation_image) * self.cfg.tactile_depth_enhance_scale  # scale the depth difference to [0, 1] (no clamp)
                img = (cam_img * 255).to(torch.uint8)

            if "shear" in self.cfg.tactile_image_type:
                if self.cfg.if_depth:
                    cam_img_original = self._tactile_camera.data.output["distance_to_image_plane"]
                    cam_img = torch.clamp(torch.abs(cam_img_original - self.no_deformation_image) * self.cfg.tactile_depth_enhance_scale , 0, 1.0)  # scale the depth difference to [0, 1] (also clamp it if exceeds 1.0)
                    # cam_img = torch.abs(cam_img_original - self.no_deformation_image) * self.cfg.tactile_depth_enhance_scale  # scale the depth difference to [0, 1] (no clamp)
                    img = (cam_img * 255).to(torch.uint8)
                motion = self._tactile_camera.data.output["motion_vectors"]  # (B, H, W, 2)
                # Create dark background
                if self.cfg.if_shear_hsv: 
                    batch_shear_hsv_imgs, hsv_img = self._visualize_shear_hsv(motion)
                if self.cfg.if_shear_arrows:
                    batch_shear_arrow_imgs, arrow_img = self._visualize_shear_sparse_arrows(motion)

            if self.cfg.tactile_image_type == 'rgb':
                cam_img = self._tactile_camera.data.output["rgb"]
                img = cam_img.to(torch.uint8)

        if img is None:
            raise ValueError(f"Unknown image_type: {self.cfg.tactile_image_type}")

        if self.cfg.if_render_tactile:
            self._render_closed = False
            if not self._render_closed:
                if self.cfg.if_depth:
                    self._render_tactile_img(img[0], title="tactile_window_depth")
                if "shear" in self.cfg.tactile_image_type and self.cfg.if_shear_hsv:
                    self._render_tactile_img(hsv_img[0], title="tactile_window_hsv")
                if "shear" in self.cfg.tactile_image_type and self.cfg.if_shear_arrows:
                    self._render_tactile_img(arrow_img[0], title="tactile_window_arrows")
        # return img.float() / 255.0
        return img

    def _visualize_shear_sparse_arrows_tensor(self, motion, step=32, scale=10, min_mag=0.01, tip_len=0.3):
        """
        Visualize motion vectors as sparse arrows with white roots.
        
        Args:
            motion: torch.Tensor or np.ndarray of shape [H, W, 2]
            step: stride for sampling (higher = fewer arrows)
            scale: length scale for arrows
            min_mag: minimum magnitude to display
        Returns:
            img: np.ndarray (H, W, 3) with arrows drawn
        """
        if isinstance(motion, np.ndarray):
            motion = torch.from_numpy(motion)
        device = motion.device if motion.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        motion = motion.to(device)

        H, W, _ = motion.shape
        dx, dy = motion[..., 0], motion[..., 1]

        # --- sparse sampling grid ---
        ys, xs = torch.meshgrid(
            torch.arange(0, H, step, device=device),
            torch.arange(0, W, step, device=device),
            indexing="ij",
        )
        vx = dx[ys, xs]
        vy = dy[ys, xs]
        mag = torch.sqrt(vx**2 + vy**2)

        # --- background ---
        img = torch.full((H, W, 3), 40, dtype=torch.uint8, device=device)

        # --- draw white roots ---
        img[ys, xs] = torch.tensor([255, 255, 255], dtype=torch.uint8, device=device)

        # --- valid motion mask ---
        mask = mag >= min_mag
        if not mask.any():
            return img.cpu().numpy()

        xs_s, ys_s = xs[mask].float(), ys[mask].float()
        vx_s, vy_s = vx[mask], vy[mask]

        # --- arrow endpoints ---
        x2 = (xs_s + vx_s * scale).round().clamp(0, W - 1).long()
        y2 = (ys_s + vy_s * scale).round().clamp(0, H - 1).long()

        # --- draw main shaft (green line) ---
        n_steps = 10
        t = torch.linspace(0, 1, n_steps, device=device).view(1, -1)
        x_line = (xs_s.unsqueeze(1) + (x2 - xs_s).unsqueeze(1) * t).long()
        y_line = (ys_s.unsqueeze(1) + (y2 - ys_s).unsqueeze(1) * t).long()
        img[y_line, x_line] = torch.tensor([0, 255, 0], dtype=torch.uint8, device=device)

        # --- draw arrowheads (two short angled lines) ---
        # compute direction unit vectors
        dx_dir = x2.float() - xs_s
        dy_dir = y2.float() - ys_s
        mag_dir = torch.sqrt(dx_dir**2 + dy_dir**2 + 1e-8)
        dx_dir /= mag_dir
        dy_dir /= mag_dir

        # rotate direction by ±angle for the two tip sides
        angle = torch.deg2rad(torch.tensor(30.0, device=device))
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        # rotation matrices
        dx1 =  cos_a * dx_dir + sin_a * dy_dir
        dy1 = -sin_a * dx_dir + cos_a * dy_dir
        dx2 =  cos_a * dx_dir - sin_a * dy_dir
        dy2 =  sin_a * dx_dir + cos_a * dy_dir

        tip_scale = scale * tip_len
        x_tip1 = (x2 - dx1 * tip_scale).round().clamp(0, W - 1).long()
        y_tip1 = (y2 - dy1 * tip_scale).round().clamp(0, H - 1).long()
        x_tip2 = (x2 - dx2 * tip_scale).round().clamp(0, W - 1).long()
        y_tip2 = (y2 - dy2 * tip_scale).round().clamp(0, H - 1).long()

        # draw both sides of the arrowhead
        n_tip = 5
        t_tip = torch.linspace(0, 1, n_tip, device=device).view(1, -1)
        for (x_start, y_start, x_end, y_end) in [
            (x2, y2, x_tip1, y_tip1),
            (x2, y2, x_tip2, y_tip2),
        ]:
            x_head = (x_start.unsqueeze(1) + (x_end - x_start).unsqueeze(1) * t_tip).long()
            y_head = (y_start.unsqueeze(1) + (y_end - y_start).unsqueeze(1) * t_tip).long()
            img[y_head, x_head] = torch.tensor([0, 255, 0], dtype=torch.uint8, device=device)

        # --- move once to CPU for OpenCV display ---
        return img.cpu().numpy()

    def _visualize_shear_sparse_arrows(self, motion, step=12, scale=15, min_mag=0.01):
        """
        Visualize motion vectors as sparse arrows with white root dots arranged in a circular region.

        Args:
            motion: torch.Tensor or np.ndarray, shape [B, H, W, 2] or [H, W, 2]
            step: sampling stride (higher = fewer arrows)
            scale: arrow length scale
            min_mag: min magnitude threshold to draw an arrow

        Returns:
            img_torch: torch.Tensor [B, H, W, 3], dtype=torch.uint8 (for GPU/logging)
            img_bgr: np.ndarray [H, W, 3], dtype=uint8 (for OpenCV display)
        """

        # --- Ensure torch tensor ---
        if isinstance(motion, np.ndarray):
            motion = torch.from_numpy(motion)

        if motion.ndim == 3:
            motion = motion.unsqueeze(0)  # [1, H, W, 2]
        assert motion.shape[-1] == 2, f"Expected [..., 2], got {motion.shape}"

        device = motion.device if motion.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        motion = motion.to(device)

        B, H, W, _ = motion.shape

        # --- Background tensor (torch) ---
        img_torch = torch.zeros((B, H, W, 3), dtype=torch.uint8, device=device)
        img_torch[:] = 40  # dark gray

        # --- Convert first frame to NumPy for drawing arrows ---
        motion_np = motion[0].detach().cpu().numpy()
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:] = 40  # dark gray

        dx, dy = motion_np[..., 0], motion_np[..., 1]

        # --- Define circular mask ---
        cy, cx = H // 2, W // 2
        radius = int(min(H, W) * 0.48)
        radius2 = radius * radius

        # --- Draw white dots + arrows ---
        for y in range(0, H, step):
            for x in range(0, W, step):
                if (x - cx) ** 2 + (y - cy) ** 2 > radius2:
                    continue

                # Always draw root dot
                cv2.circle(img, (x, y), 1, (255, 255, 255), -1)

                vx, vy = dx[y, x], dy[y, x]
                mag = np.hypot(vx, vy)
                if mag < min_mag:
                    continue

                x2 = int(x + vx * scale)
                y2 = int(y + vy * scale)

                cv2.arrowedLine(img, (x, y), (x2, y2), (0, 255, 0), 1, tipLength=0.3)

        # --- Convert cv2 image back to Torch tensor for consistency ---
        img_torch[0] = torch.from_numpy(img).to(torch.uint8).to(device)
        img = np.expand_dims(img, axis=0)

        return img_torch, img

    def _visualize_shear_hsv(self, motion, max_shear_mag=1.0, dynamic_norm=False):
        """
        Convert motion vector field (dx, dy) -> HSV torch tensor and BGR numpy image.

        Args:
            motion: torch.Tensor or np.ndarray of shape [B, H, W, 2] or [H, W, 2]
            max_shear_mag: float, static normalization upper bound
            dynamic_norm: bool, if True normalize per frame by its max magnitude

        Returns:
            hsv_torch: torch.Tensor [B, H, W, 3], dtype=torch.uint8
            img_bgr: np.ndarray [H, W, 3], dtype=uint8, ready for cv2.imshow()
        """
        # --- ensure torch tensor ---
        if isinstance(motion, np.ndarray):
            motion = torch.from_numpy(motion)

        if motion.ndim == 3:  # single frame [H,W,2]
            motion = motion.unsqueeze(0)  # [1,H,W,2]

        assert motion.shape[-1] == 2, f"Expected [...,2], got {motion.shape}"

        device = motion.device if motion.is_cuda else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        motion = motion.to(device)

        B, H, W, _ = motion.shape
        dx, dy = motion[..., 0], motion[..., 1]

        # --- magnitude and angle ---
        magnitude = torch.sqrt(dx ** 2 + dy ** 2)
        angle = torch.atan2(dy, dx)  # [-pi, pi]

        # --- hue [0,180] for direction ---
        hue = (angle + torch.pi) * (180.0 / (2 * torch.pi))

        # --- normalize magnitude to [0,255] ---
        if dynamic_norm:
            mag_max = torch.clamp(magnitude.flatten(1).max(dim=1)[0].view(B, 1, 1), min=1e-6)
            mag_norm = (magnitude / mag_max) * 255.0
        else:
            magnitude_clip = torch.clamp(magnitude, 0.0, max_shear_mag)
            mag_norm = (magnitude_clip / max_shear_mag) * 255.0

        # --- saturation fixed at 255 ---
        sat = torch.full_like(hue, 255.0)

        # --- stack HSV channels ---
        hsv_torch = torch.stack((hue, sat, mag_norm), dim=-1).clamp(0, 255).to(torch.uint8)

        # --- convert first frame to CPU numpy for cv2 plotting ---
        hsv_np = hsv_torch[0].cpu().numpy()  # [H,W,3], uint8

        # --- HSV → BGR ---
        img_bgr = cv2.cvtColor(hsv_np, cv2.COLOR_HSV2BGR)
        img_bgr = np.expand_dims(img_bgr, axis=0)

        return hsv_torch, img_bgr

    def _render_tactile_img(self, img, title="tactile_window"):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        # Convert RGB -> BGR for OpenCV
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Resize to 512x512
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)  # TODO: Fixed sized for visualization.
        if not self._render_closed:
            cv2.imshow(title, img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to close
                cv2.destroyWindow(title)
                self._render_closed = True

    def _get_observations(self) -> dict:
        teacher_obs = torch.clamp(self._get_oracle_obs(), -5.0, 5.0)  # TODO: Finetune the clamp value (for oracle, max=1, min=-1 due to positional encoding. For tactile image? ). This clamping is for numerical stability
        if self.cfg.if_save_tactile_reference_image:
            self._save_tactile_reference_images()
        elif not self._if_loaded_tactile_reference_image:  # load only once
            self._if_loaded_tactile_reference_image = True
            self._load_tactile_reference_images()
        student_obs = self._get_tactile_images_tensors()
        student_obs = student_obs.permute(0, 3, 1, 2)  # [B, H, W, C] → [B, C, H, W]
        img = torch.clamp(student_obs, -5.0, 5.0)
        dummy_1d = torch.zeros((img.shape[0], 1), device=img.device)

        return {"image": img, "dummy_1d": dummy_1d, "oracle": teacher_obs}  # TODO: Finetune the clamp value (for oracle, max=1, min=-1 due to positional encoding. For tactile image? ). This clamping is for numerical stability
    
    def _positional_encoding(self, x: torch.Tensor, num_freqs: int = 4):
        """
        Apply sinusoidal positional encoding to input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (N, D)
            num_freqs (int): Number of frequency bands
        Returns:
            torch.Tensor: Encoded tensor of shape (N, D * 2 * num_freqs)
        """
        N, D = x.shape
        freq_bands = 2 ** torch.arange(num_freqs, dtype=torch.float32, device=x.device) * torch.pi
        x = x.unsqueeze(-1)  # (N, D, 1)
        freqs = freq_bands.view(1, 1, -1)  # (1, 1, F)

        x_freq = x * freqs  # (N, D, F)
        sin_x = torch.sin(x_freq)
        cos_x = torch.cos(x_freq)

        return torch.cat([sin_x, cos_x], dim=-1).view(N, D * 2 * num_freqs)  # (N, D * 2F)

    def _rpy_from_quat(self, quat: torch.Tensor):  # TODO: have another class or util file for these conversion functions
        '''
        Convert quaternion to roll, pitch, yaw. Batch version.
        '''
        return torch.stack(euler_xyz_from_quat(quat), dim=-1)

    def _quat_from_rpy(self, rpy: torch.Tensor):  # TODO: have another class or util file for these conversion functions
        '''
        Convert roll, pitch, yaw to quaternion. Batch version.
        '''
        return quat_from_euler_xyz(*rpy.unbind(-1))

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
        # print("distance_from_edge:", distance_from_edge)
        # print("dist_to_goal:", dist_to_goal)
        # This reward cannot learn well
        terminal_penalty = -300.0

        # def reward_shaping_distance(d: torch.Tensor) -> torch.Tensor:
        #     """Reward shaping based on distance to goal."""
        #     dist_reward = 1.0 / (1.0 + d**2)
        #     dist_reward *= dist_reward
        #     dist_reward = torch.where(d <= 0.01, dist_reward * 2, dist_reward)
        #     return dist_reward
        
        # distance_from_edge = -reward_shaping_distance(distance_from_edge)
        # dist_to_goal = -reward_shaping_distance(dist_to_goal)
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