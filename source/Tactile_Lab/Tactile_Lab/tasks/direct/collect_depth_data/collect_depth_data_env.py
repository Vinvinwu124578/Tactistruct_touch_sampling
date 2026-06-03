# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import warnings
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
from .collect_depth_data_env_cfg import CollectDepthDataEnvCfg 
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors.camera import TiledCamera
import cv2
import numpy as np
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, euler_xyz_from_quat, quat_unique, quat_slerp

from Tactile_Lab.utility.tactile_sensor import TactileSensor
from Tactile_Lab.utility.utils import positional_encoding, rpy_from_quat, quat_from_rpy, make_dir, save_json_obj, quat_slerp_batch
from Tactile_Lab.utility.setup_targets import setup_targets, read_target_df_csv
from Tactile_Lab.utility.setup_targets import POSE_LABEL_NAMES, SHEAR_LABEL_NAMES, OBJECT_POSE_LABEL_NAMES

class CollectDepthDataEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: CollectDepthDataEnvCfg

    def __init__(self, cfg: CollectDepthDataEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.target_pose = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)

        self.counter = 0
        # #################################### For IK ####################################
        self.diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")  # TODO: set the use_relative_mode for RL training
        self.diff_ik_controller = DifferentialIKController(self.diff_ik_cfg, num_envs=self.num_envs, device=self.device)
        # Markers
        if self.cfg.if_debug:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self.ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
            self.obj_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/obj_current"))
        # Define workframe and initial end-effector pose
        self.workframe_scene = torch.tensor(self.cfg.workframe_scene).unsqueeze(0).repeat(self.num_envs, 1).to(self.device)
        self.workframe_quat_scene = quat_from_rpy(self.workframe_scene[:, 3:6])
        self.workframe_scene = torch.cat([self.workframe_scene[:, 0:3], self.workframe_quat_scene], dim=-1)
        self.workframe_world = self.workframe_scene.clone()
        self.workframe_world[:, 0:3] += self.scene.env_origins[:, 0:3]
        self.ee_init_pose_world = self.workframe_scene[0].clone()  # (7,), use the first env's workframe as the initial ee pose in world frame, which will be used as the default target pose for data collection
        self.ee_init_pose_world = torch.tensor(self.ee_init_pose_world, device=self.device)
        # Track the given command
        # Create buffers to store actions
        self.ik_commands = torch.zeros(self.num_envs, self.diff_ik_controller.action_dim, device=self._robot.device)
        self.ik_commands[:] = self.ee_init_pose_world
        if self.cfg.robot_name == "ur5_tactip":
            self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["tcp_link"])
        self.robot_entity_cfg.resolve(self.scene)
        # Obtain the frame index of the end-effector
        # For a fixed base robot, the frame index is one less than the body index. This is because
        # the root body is not included in the returned Jacobians.
        if self._robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

        # setup date colletion
        BASE_DATA_PATH = os.path.dirname(os.path.abspath(__file__))
        if self.cfg.target_dir is None:
            self.save_dir = os.path.join(BASE_DATA_PATH, 'test')
            self.image_dir = os.path.join(self.save_dir, "sensor_images")
            make_dir(self.save_dir)
            make_dir(self.image_dir)
            self.target_df = setup_targets(
                self.cfg.collect_params,
                self.cfg.num_poses,
                self.save_dir
            )
        else:
            self.save_dir = os.path.join(BASE_DATA_PATH,
                                        'data/sim/',
                                        self.cfg.tactile_image_type,
                                        self.cfg.tap_or_shear,
                                        self.cfg.target_dir_name
                                    )
            self.image_dir = os.path.join(self.save_dir, "images")
            make_dir(self.save_dir)
            make_dir(self.image_dir)
            og_collect_dir = os.path.join(self.cfg.target_home_dir, self.cfg.target_dir_name)
            og_target_file = os.path.join(og_collect_dir, 'targets.csv')
            self.target_df = read_target_df_csv(og_target_file, self.cfg.shuffle_data, self.save_dir)

        save_json_obj(self.cfg.sensor_params, os.path.join(self.save_dir, 'sensor_params'))

    def _setup_scene(self):
        self.obs_type = self.cfg.obs_type
        self._robot = Articulation(self.cfg.robot_cfg)
        self._object = RigidObject(self.cfg.edge_cfg)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["edge"] = self._object
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

    def step_sim(self):
        # perform physics stepping
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        self._update_robot_kinematics()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
        if self.cfg.if_debug:
            ee_pose_w = self.ee_pose_world
            self.ee_marker.visualize(
                ee_pose_w[:, 0:3],
                ee_pose_w[:, 3:7],
            )
            self.obj_marker.visualize(self._object.data.root_pos_w, self._object.data.root_quat_w)
                                      
    def convert_pose_from_rpy_to_quat(self, actions: torch.Tensor):
        """
        Use actions directly as target pose.
        actions = [x, y, z, roll, pitch, yaw]
        """
        target_pos = actions[:, 0:3]
        target_quat = quat_from_rpy(actions[:, 3:6])
        target_pose = torch.cat([target_pos, target_quat], dim=-1)
        return target_pose

    def _update_robot_kinematics(self):
        self.ee_pose_world = self._robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]].clone()
        robot_root_pose_world = self._robot.data.root_pose_w.clone()

        self.ee_pos_b, self.ee_quat_b = subtract_frame_transforms(
            robot_root_pose_world[:, 0:3], robot_root_pose_world[:, 3:7],
            self.ee_pose_world[:, 0:3], self.ee_pose_world[:, 3:7]
        )

    def set_target_tcp_pose(self, cur_target_pose: torch.Tensor):
        # obtain quantities from simulation
        self._update_robot_kinematics()
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
        # self.actions = cur_target_pose
        # self.scaled_actions = (
        #     self.robot_ee_speed_scale
        #     * self.dt
        #     * self.actions
        #     * self.cfg.action_scale
        # )
        # self.checked_scaled_actions = self.check_TCP_vel_lims(self.actions)
        # set_trace()
        # set_trace()
        # self.prev_targets[:] = self.cur_targets_pos_rpy.clone()

        self.ik_commands = cur_target_pose
        self.diff_ik_controller.set_command(self.ik_commands)

        self.robot_dof_targets = self.diff_ik_controller.compute(self.ee_pos_b, self.ee_quat_b, jacobian, joint_pos)

    def blocking_position_move(
        self,
        max_steps=1000,
        constant_vel=None,
        j_pos_tol=0.1,
        j_vel_tol=0.1,
    ):
        
        # self._robot.set_joint_position_target(self.robot_dof_targets, joint_ids=self.robot_entity_cfg.joint_ids)
        # target joint positions (tensor)
        targ_j_pos = self.robot_dof_targets.clone()

        for i in range(max_steps):
            if i % 100 == 0:
                print(f"Step {i}: Moving joints to target positions with blocking_position_move...")
            # current joint state
            cur_j_pos = self._robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
            cur_j_vel = self._robot.data.joint_vel[:, self.robot_entity_cfg.joint_ids]

            if constant_vel is not None and constant_vel > 0:

                diff_j = targ_j_pos - cur_j_pos
                norm = torch.linalg.norm(diff_j, dim=-1, keepdim=True)
                # v: unit vector pointing toward the target, which only encodes the direction in joint space, not the magnitude of the error.
                v = torch.where(
                    norm > 0,
                    diff_j / (norm + 1e-8),
                    torch.zeros_like(diff_j),
                )

                step_j = cur_j_pos + v * constant_vel

                # reduce velocity when close to target
                if torch.all(torch.abs(diff_j) < constant_vel):
                    constant_vel *= 0.5

            else:
                step_j = targ_j_pos

            # send command to IsaacLab articulation
            joint_vel = torch.zeros_like(targ_j_pos)

            self._robot.set_joint_position_target(
                step_j,
                joint_ids=self.robot_entity_cfg.joint_ids
            )
            # self._robot.write_joint_state_to_sim(
            #     position=step_j,
            #     velocity=joint_vel,
            #     joint_ids=self.robot_entity_cfg.joint_ids,
            # )
            self._robot.reset()
            # step simulation
            self.step_sim()

            # compute errors
            j_pos_err = torch.sum(torch.abs(targ_j_pos - cur_j_pos))
            j_vel_err = torch.sum(torch.abs(cur_j_vel))
            if i % 100 == 0:
                print(f"Step {i}: Joint position error: {j_pos_err:.6f}, Joint velocity error: {j_vel_err:.6f}")
            if (j_pos_err < j_pos_tol) and (j_vel_err < j_vel_tol):
                break
            
        if i == max_steps - 1:
            warnings.warn(
                "Blocking position move failed to reach tolerance within max_steps."
            )

    def apply_blocking_position_move(self, quick_mode=False):
        percentage = 100
        self._min_constant_vel = 0.01
        self._max_constant_vel = 0.1
        constant_vel_range = self._max_constant_vel - self._min_constant_vel
        self._constant_vel = self._min_constant_vel + constant_vel_range * (percentage / 100.0)
        self._max_position_move_steps = 100000

        if not quick_mode:
            print("Applying blocking position move with constant velocity:")
            # slow but more realistic moves
            self.blocking_position_move(
                max_steps=self._max_position_move_steps,
                constant_vel=self._constant_vel,
                j_pos_tol=0.001,
                j_vel_tol=1e-3,
            )

        else:
            # fast but unrealistic moves (bigger_moves = worse performance)
            self.blocking_position_move(
                max_steps=1000,
                constant_vel=None,
                j_pos_tol=1e-6,
                j_vel_tol=1e-3,
            )

    def move_linear(self, targ_pose, quick_mode=False):
        """
        Move the robot linearly to the target pose.

        Args:
            targ_pose (torch.Tensor): The target pose in the base frame with position (x, y, z) and orientation (rpy or quat).
            quick_mode (bool): If True, use fast but unrealistic moves. Defaults to False.
        """
        # Detect pose format automatically
        if targ_pose.shape[-1] == 6:  # (x, y, z, r, p, y)
            targ_pose = self.convert_pose_from_rpy_to_quat(targ_pose)
        elif targ_pose.shape[-1] == 7:  # (x, y, z, qx, qy, qz, qw)
            pass
        else:
            raise ValueError(
                f"targ_pose must have 6 (RPY) or 7 (quat) elements, got shape {targ_pose.shape}"
            )

        self._update_robot_kinematics()
        self._max_outer_loop_steps = 10
        self.target_pose = targ_pose
        alpha_pos = 1
        alpha_rot = 1

        for i in range(self._max_outer_loop_steps):

            step_pos = self.ee_pos_b + alpha_pos * (targ_pose[:, :3] - self.ee_pos_b)

            target_quat = targ_pose[:, 3:7]
            step_quat = quat_slerp_batch(self.ee_quat_b, target_quat, alpha_rot)
            step_quat = F.normalize(step_quat, dim=-1)

            step_pose = torch.cat([step_pos, step_quat], dim=-1)

            self.set_target_tcp_pose(step_pose)
            self.apply_blocking_position_move(quick_mode=quick_mode)
            self._update_robot_kinematics()
            target_pose_err, axis_angle_error = compute_pose_error(targ_pose[:,:3], targ_pose[:,3:7], self.ee_pos_b, self.ee_quat_b)
            print(
                f"Outer loop step {i}: "
                f"Target pose error: {target_pose_err.mean().item():.6f}, "
                f"Axis-angle error: {axis_angle_error.mean().item():.6f}"
            )
            if torch.all(target_pose_err.abs() < 1e-3) and torch.all(axis_angle_error.abs() < 1e-3):
                print(f"Convergence achieved at outer loop step {i}.")
                break
        if i == self._max_outer_loop_steps - 1:
            warnings.warn(
                "Max outer loop steps reached in move_linear without convergence."
            )

    def _pre_physics_step(self, actions: torch.Tensor):

        actions = F.pad(actions, pad=(0, 4), mode='constant', value=0)  # (N, 6)
        # set_trace()
        if hasattr(self, "actions"):
            self.pre_actions = self.actions.clone()
            if self.pre_actions.shape[1] == 2:
                self.pre_actions = F.pad(self.pre_actions.clone(), pad=(0, 4), mode='constant', value=0)  # (N, 6)
        else:
            self.pre_actions = torch.zeros_like(actions, device=self.device)
        self.counter +=1

        # actions[:,0] = 0.750000
        # actions[:,2] = 0.322000


        # self.move_linear(actions)
        self.collect_data(
            targets_df=self.target_df,
            image_dir=self.image_dir,
            collect_params=self.cfg.collect_params,
        )

    def _apply_action(self):
        pass
        # self._robot.set_joint_position_target(self.robot_dof_targets, joint_ids=self.robot_entity_cfg.joint_ids)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        # No termination conditions used
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Only truncate based on episode length
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _compute_intermediate_values(self):
        self.ee_pose_world = self._robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]].clone()
        print("ee_pose_world:", self.ee_pose_world)
        print("self.target_pos:", self.target_pose)
        self.ee_pose_scene = self.ee_pose_world[:, 0:3] - self.scene.env_origins

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
        return self._compute_rewards()

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        ## Reset the robot to the default joint positions, and then reset it to the ee_init_pose_world!
        joint_pos_env_ids = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel_env_ids = self._robot.data.default_joint_vel[env_ids].clone()

        self._robot.set_joint_position_target(joint_pos_env_ids, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos_env_ids, joint_vel_env_ids, env_ids=env_ids)
        self._robot.reset(env_ids)  # TODO: Is it necessary to call reset?? Some example in the IsaacLab does not call reset after setting joint states.

        ## Now reset it to the ee_init_pose_world!
        # reset actions
        self.ik_commands[env_ids, :] = self.ee_init_pose_world.clone()
        self.prev_targets[env_ids, :] = self.ee_init_pose_world.clone()
        self.target_pose[env_ids, :] = self.ee_init_pose_world.clone()
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

        robot_root_pose_world = self._robot.data.root_pose_w.clone()
        self.workframe_pos_base, self.workframe_quat_base = subtract_frame_transforms(
            robot_root_pose_world[:, 0:3], robot_root_pose_world[:, 3:7], self.workframe_world[:, 0:3], self.workframe_world[:, 3:7]
        )

        # reset edge object
        # compute intermediate values after reset
        self._compute_intermediate_values()

    def _get_oracle_obs(self) -> dict[str, torch.Tensor]:
        """
        Get the oracle observations. All in world frame. #TODO: Change to workframe when deploy to real robot
        """

        pass
    
    def _get_observations(self) -> dict:

        dummy_1d = torch.zeros((1, 1), device=self.device)

        return {
            "dummy_1d": dummy_1d   # MLP input (B, 1)
        }

    def _compute_rewards(self):
        # data collection environment does not use rewards.
        # Return a zero reward for all envs.
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    def save_tactile_image(self, image_outfile):
        """Collect and save reference tactile image."""
        if self.tactile_sensor_cfg.if_save_tactile_reference_image:
            self.tactile_sensor._save_tactile_reference_images()
        elif not self._if_loaded_tactile_reference_image:  # load only once
            self.tactile_sensor._load_tactile_reference_images()
            self._if_loaded_tactile_reference_image = True
        tactile_obs = self.tactile_sensor._get_tactile_images_tensors()
        if isinstance(tactile_obs, tuple):
            tactile_obs = tactile_obs[0]
        img = tactile_obs[0].detach()
        if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
            img = img.permute(1, 2, 0)
        if img.ndim == 3 and img.shape[-1] == 2:
            img = img[..., 0]
        if img.dtype.is_floating_point:
            img_np = (torch.clamp(img, 0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
        else:
            img_np = torch.clamp(img, 0, 255).cpu().numpy().astype(np.uint8)
        if img_np.ndim == 3 and img_np.shape[-1] == 1:
            img_np = img_np[..., 0]
        cv2.imwrite(image_outfile, img_np)

    def workframe_to_base(self, pose_work):
        pos_work = pose_work[:, :3]
        quat_work = quat_from_rpy(pose_work[:, 3:6])

        # robot_root_pose_world = self._robot.data.root_pose_w.clone()
        # self.workframe_pos_base, self.workframe_quat_base = subtract_frame_transforms(
        #     robot_root_pose_world[:, 0:3], robot_root_pose_world[:, 3:7], self.workframe_world[:, 0:3], self.workframe_world[:, 3:7]
        # )
        pos_base, quat_base = combine_frame_transforms(
            self.workframe_pos_base,
            self.workframe_quat_base,
            pos_work,
            quat_work
        )
        rpy_base = rpy_from_quat(quat_base)
        return torch.cat([pos_base, rpy_base], dim=-1)

    def collect_data(
        self,
        targets_df,
        image_dir,
        collect_params,
    ):
        pose_label_names = collect_params.get('pose_label_names', POSE_LABEL_NAMES)
        shear_label_names = collect_params.get('shear_label_names', SHEAR_LABEL_NAMES)
        # object_pose_label_names = collect_params.get('object_pose_label_names', OBJECT_POSE_LABEL_NAMES)

        # start 50mm above workframe origin with zero joint 6
        pose_work = torch.tensor([[0, 0, -0.05, 0, 0, 0]], device=self.device)
        pose_baseframe_hover = self.workframe_to_base(pose_work)
        self.move_linear(pose_baseframe_hover)
        # image_outfile = os.path.join(image_dir, 'image_0.png')
        # self.save_tactile_image(image_outfile)

        # clear object by 10mm
        clearance = torch.tensor([[0, 0, 0.01, 0, 0, 0]], device=self.device)
        pose_baseframe_reset = self.workframe_to_base(torch.zeros((1, 6), device=self.device) - clearance)
        # self.move_linear(pose_base_reset)
        saved_obj_label = ''

        # ==== data collection loop ====
        for i, row in targets_df.iterrows():
            print(f"Collecting data for target {i+1}/{len(targets_df.index)}...")
            image_name = row.loc["sensor_image"]
            if "obj_id" in row.index:
                obj_label = row.loc["obj_id"]
            elif "object_label" in row.index:
                obj_label = row.loc["object_label"]
            else:
                obj_label = "default"

            pose_np = row.loc[pose_label_names].values.astype(float)
            shear_np = row.loc[shear_label_names].values.astype(float)
            # obj_pose_np = row.loc[object_pose_label_names].values.astype(float)

            # Convert RPY (deg → rad)
            pose_np[0:3] /= 1000.0
            shear_np[0:3] /= 1000.0
            # obj_pose_np[0:3] /= 1000.0
            pose_np[3:6] = np.deg2rad(pose_np[3:6])
            shear_np[3:6] = np.deg2rad(shear_np[3:6])
            # print("testing pose oorientation to be 0")
            # pose_np[3:6] = 0
            # obj_pose_np[3:6] = np.deg2rad(obj_pose_np[3:6])

            pose = torch.tensor(
                pose_np, dtype=torch.float32, device=self.device
            )[None, :].expand(self.num_envs, -1)

            shear = torch.tensor(
                shear_np, dtype=torch.float32, device=self.device
            )[None, :].expand(self.num_envs, -1)

            # obj_pose = torch.tensor(
            #     obj_pose_np, dtype=torch.float32, device=self.device
            # )[None, :].expand(self.num_envs, -1)
            # TODO: Ignore object pose
            obj_pose = torch.zeros(
                (self.num_envs, 6),
                dtype=torch.float32,
                device=self.device,
            )
            # report
            with np.printoptions(precision=1, suppress=True):
                print(f"{i+1}/{len(targets_df.index)}: [{obj_label}] pose{pose}, shear{shear}")

            # new object set reset
            if obj_label != saved_obj_label:
                saved_obj_label = obj_label
                self.move_linear(pose_baseframe_reset)
                pose_baseframe_reset = self.workframe_to_base(obj_pose - clearance)
                self.move_linear(pose_baseframe_reset)
                # joint_angles = robot.joint_angles
    
            # pose is relative to object pose
            
            pose += obj_pose

            # move to above new pose (avoid changing pose in contact with object)
            move_pose_baseframe = self.workframe_to_base(pose + shear - clearance)
            self.move_linear(move_pose_baseframe)

            # move down to offset pose
            move_pose_baseframe = self.workframe_to_base(pose + shear)
            self.move_linear(move_pose_baseframe)

            # move to target pose inducing shear
            move_pose_baseframe = self.workframe_to_base(pose)
            self.move_linear(move_pose_baseframe)

            # collect and process tactile image
            image_outfile = os.path.join(image_dir, image_name)
            self.save_tactile_image(image_outfile)

            # move above the target pose
            move_pose_baseframe = self.workframe_to_base(pose - clearance)
            self.move_linear(move_pose_baseframe)

            # if sorted, don't move to reset position
            if not collect_params.get('sort', False):
                self.move_linear(pose_baseframe_reset)
            print(f"Finished collecting data for target {i+1}/{len(targets_df.index)}.\n")

        # finish 50mm above workframe origin then zero last joint
        self.move_linear(pose_baseframe_hover)
        self.move_linear(pose_baseframe_reset)
