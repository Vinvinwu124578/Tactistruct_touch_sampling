# hand_gait_env.py
import os
import torch
import numpy as np
from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_mul, quat_from_angle_axis, sample_uniform, axis_angle_from_quat, quat_rotate_inverse, quat_apply_inverse, combine_frame_transforms
from isaaclab.sensors import ContactSensor
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
from Tactile_Lab.tasks.direct.hand_gait.hand_gait_env_cfg import HandGaitEnvCfg
import isaaclab.sim as sim_utils
from ipdb import set_trace
from pxr import UsdGeom
from omni.usd import get_context
import isaaclab.sim as sim_utils

from isaaclab.utils.math import (
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    sample_uniform,
    saturate,
    quat_from_euler_xyz,
    euler_xyz_from_quat,
)
from Tactile_Lab.utility.tactile_sensor import TactileSensor
from Tactile_Lab.utility.utils import quat_from_rpy, rpy_from_quat, positional_encoding
from isaaclab.sensors.camera import TiledCamera

torch.set_printoptions(precision=4, sci_mode=False)


class HandGaitEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: HandGaitEnvCfg

    def __init__(self, cfg: HandGaitEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # DOFs
        self.num_hand_dofs = self.hand.num_joints

        # Fingertips
        self.finger_bodies = torch.tensor(
            [self.hand.body_names.index(n) for n in cfg.fingertip_body_names],
            device=self.device,
            dtype=torch.long,
        )

        # buffers for position targets
        self.hand_dof_cur_pos = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_hand_dof_pos = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.hand_dof_cur_vel = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[0, :, 0]
        self.hand_dof_upper_limits = joint_pos_limits[0, :, 1]
        
        # obj buffers
        self.object_position_sceneframe = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.object_position_prev_sceneframe = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.object_rot = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)
        self.object_rot_prev = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)
        self.object_rotate_velocity = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float)
        # --- domain randomization buffers ---
        self.object_mass = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float)
        self.object_static_friction = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float)
        self.object_dynamic_friction = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float)
        self.object_restitution = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float)
        self.object_com = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.object_anchor_pos_sceneframe = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.object_disturb_force = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.object_disturb_torque = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.object_pose_in_handframe = ( torch.tensor(self.cfg.object_pose_in_handframe, device=self.device) .unsqueeze(0) .repeat(self.num_envs, 1) )
        # reward buffers
        self.rot_axis_buf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.init_pose_buf = torch.zeros((self.num_envs, self.num_hand_dofs), device=self.device, dtype=torch.float)

        if self.cfg.if_random_joint_impulse:
            self._impulse_timer = torch.zeros(self.num_envs, device=self.device)
            self._impulse_joint = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._impulse_prob = 0.01        # chance to start impulse
            self._impulse_duration = 300     # consecutive steps
            self._impulse_strength = 10.0    # magnitude

        # Load cached grasps
        assert os.path.exists(self.cfg.grasp_cache_path), f"Grasp cache not found: {self.cfg.grasp_cache_path}"

        grasp_data = np.load(self.cfg.grasp_cache_path)  # (N, 23)
        self.grasp_data = torch.tensor(
            grasp_data, device=self.device, dtype=torch.float
        )

        self.num_grasps = self.grasp_data.shape[0]
        # ---- observation history config ----
        self.single_prop_dim = self.num_hand_dofs * 2  # (pos + target)

        # ---- buffers ----
        self.proprio_buf_lag_history = torch.zeros(
            self.num_envs,
            self.cfg.obs_hist_total_len,
            self.single_prop_dim,
            device=self.device,
        )

        if self.cfg.if_use_low_dim_contact_obs:
            self.contact_dim = len(self.fingertip_contact_sensors) * self.cfg.contact_dim_per_finger

            self.contact_buf_lag_history = torch.zeros(
                self.num_envs,
                self.cfg.obs_hist_len,
                self.contact_dim,
                device=self.device,
            )
            self.proprio_contact_dim = self.single_prop_dim + self.contact_dim

            self.proprio_and_contact_buf_lag_history = torch.zeros(
                self.num_envs,
                self.cfg.obs_hist_total_len,
                self.proprio_contact_dim,
                device=self.device,
            )
        # ---- Depth map history ----
        if self.cfg.obs_type == "tactile":
            self.depth_channels = 1
            self.flow_channels = 2

            self.depth_map_lag_history = torch.zeros(
                self.num_envs,
                self.cfg.obs_hist_total_len,
                self.depth_channels,
                self.cfg.tactile_img_size * self.cfg.tactile_sensors_num,  # width dimension is num_sensors * img_size
                self.cfg.tactile_img_size,
                device=self.device,
            )

            # ---- Tactile flow history ----
            self.tactile_flow_lag_history = torch.zeros(
                self.num_envs,
                self.cfg.obs_hist_total_len,
                self.flow_channels,
                self.cfg.tactile_img_size * self.cfg.tactile_sensors_num,  # width dimension is num_sensors * img_size
                self.cfg.tactile_img_size,
                device=self.device,
            )

        # Debug visualization
        if self.cfg.if_frames_vis:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
            self.hand_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/hand_frame"))
            # frame_marker_cfg = FRAME_MARKER_CFG.copy()
            # frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            # self.camera_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/camera_frame"))
            self.obj_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/obj_frame"))
            # self.goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/goal_frame"))


    # ------------------------------------------------------------------ #
    # Scene
    # ------------------------------------------------------------------ #
    def _setup_scene(self):
        self.obs_type = self.cfg.obs_type
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        
        spawn_ground_plane("/World/ground", GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["hand"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        # ------------------
        # Tactile cameras
        # ------------------
        if self.obs_type == "tactile" or self.cfg.if_tactile_vis:
            self.tactile_sensors = {}
            self._if_loaded_tactile_reference_image = {}
            for cfg in self.cfg.tactile_sensor_cfgs:
                cam = TiledCamera(cfg.tactile_camera)
                self.scene.sensors[f"tactile_camera_{cfg.idx}"] = cam
                self.tactile_sensors[cfg.idx] = TactileSensor(
                    cam,
                    cfg,
                    self.device,
                    self.num_envs,
                    __file__,
                )
                self._if_loaded_tactile_reference_image[cfg.idx] = False
        # ------------------
        # Contact sensors
        # ------------------
        if self.cfg.if_use_low_dim_contact_obs:
            self.fingertip_contact_sensors = {}
            for name, sensor_cfg in self.cfg.fingertip_contact_sensors_cfgs.items():
                sensor = ContactSensor(sensor_cfg)
                self.fingertip_contact_sensors[name] = sensor
                self.scene.sensors[name] = sensor

        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)

        # --------------------------------------------------
        # Sample cached grasp poses
        # --------------------------------------------------
        num = len(env_ids)
        grasp_ids = torch.randint(
            0, self.num_grasps, (num,), device=self.device
        )
        gaits = self.grasp_data[grasp_ids]
        DIM = self.num_hand_dofs
        finger_joint_pos = gaits[:, :DIM]
        hand_pose_local = gaits[:, DIM:DIM+7]
        obj_pose_local = gaits[:, DIM+7:DIM+14]

        # Hand joint positions
        # finger_joint_pos = torch.clamp(finger_joint_pos, self.hand_dof_lower[0], self.hand_dof_upper[0])
        finger_joint_pos = saturate(
            finger_joint_pos,
            self.hand_dof_lower_limits,
            self.hand_dof_upper_limits,
        )

        self.init_pose_buf[env_ids, :] = finger_joint_pos.clone()
        # set_trace()
        self.cur_targets[env_ids] = finger_joint_pos
        self.prev_targets[env_ids] = finger_joint_pos
        self.hand.set_joint_position_target(finger_joint_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(finger_joint_pos, torch.zeros_like(finger_joint_pos), env_ids=env_ids)
        if self.cfg.ctrl_mode == "torque":
            self.hand.set_joint_effort_target(
                torch.zeros((len(env_ids), self.num_hand_dofs), device=self.device),
                env_ids=env_ids,
            )
            
        hand_pose_local = hand_pose_local.clone()
        hand_state = torch.zeros((len(env_ids), 13), device=self.device)
        hand_state[:, :3] = hand_pose_local[:, :3] + self.scene.env_origins[env_ids]
        hand_state[:, 3:7] = hand_pose_local[:, 3:7]
        hand_state[:, 7:] = 0.0
        self.hand_pose = self.hand.data.root_state_w[:, :7].clone()
        self.hand_pose[env_ids] = hand_state[:, :7]
        
        self.hand.write_root_pose_to_sim(hand_state[:, :7], env_ids)
        self.hand.write_root_velocity_to_sim(hand_state[:, 7:], env_ids)
        
        # Object
        self._update_object_physics_properties(env_ids)
        # Object pose
        obj_pose = obj_pose_local.clone()
        obj_pose[:, :3] += self.scene.env_origins[env_ids]

        self.object.write_root_pose_to_sim(obj_pose, env_ids)
        self.object.write_root_velocity_to_sim(
            torch.zeros((num, 6), device=self.device), env_ids
        )
        # self.object_position_sceneframe = self.object.data.root_pos_w - self.scene.env_origins
        self.object_position_sceneframe[env_ids] = (
            obj_pose[:, :3] - self.scene.env_origins[env_ids]
        )

        self.object_position_prev_sceneframe[env_ids] = (
            obj_pose[:, :3] - self.scene.env_origins[env_ids]
        )

        self.object_rot[env_ids] = obj_pose[:, 3:7]
        self.object_rot_prev[env_ids] = obj_pose[:, 3:7]
        self.object_rotate_velocity[env_ids] = torch.zeros((num, 1), device=self.device)
        # Since the object can fly away if using dirtubance forces, 
        # we need to have a radius distance between the hand and the object to reset condition, 
        self.object_anchor_pos_sceneframe[env_ids, :], _ = combine_frame_transforms(
            hand_pose_local[:, :3],  # the hand does not need env_id indexing since it already includes when derived it from default_root_state
            hand_pose_local[:, 3:7],
            self.object_pose_in_handframe[env_ids, :3],
            self.object_pose_in_handframe[env_ids, 3:7],
        )
        # --------------------------------------------------
        # Reset observation history (proprio)
        # --------------------------------------------------
        self.hand_dof_cur_pos[env_ids] = self.hand.data.joint_pos[env_ids]
        self.hand_dof_cur_vel[env_ids] = torch.zeros_like(self.hand_dof_cur_vel[env_ids])

        # Fill entire history with current state
        T = self.proprio_buf_lag_history.shape[1]  # 80
        self.proprio_buf_lag_history[env_ids, :, 0:16] = ( self.hand_dof_cur_pos[env_ids, None, :].expand(-1, T, -1) )
        self.proprio_buf_lag_history[env_ids, :, 16:32] = ( self.hand_dof_cur_pos[env_ids, None, :].expand(-1, T, -1) )
        if self.obs_type == "tactile":
            self.depth_map_lag_history[env_ids] = 0.0
            self.tactile_flow_lag_history[env_ids] = 0.0
        if self.cfg.if_use_low_dim_contact_obs:
            self.contact_buf_lag_history[env_ids] = 0.0
            self.proprio_and_contact_buf_lag_history[env_ids] = 0.0
        if self.cfg.if_force_disturbance:

            force_min, force_max = self.cfg.force_range
            torque_min, torque_max = self.cfg.torque_range

            self.object_disturb_force[env_ids] = torch.empty(
                len(env_ids), 3, device=self.device
            ).uniform_(force_min, force_max)

            self.object_disturb_torque[env_ids] = torch.empty(
                len(env_ids), 3, device=self.device
            ).uniform_(torque_min, torque_max)

            body_ids = torch.arange(
                self.object.num_bodies,
                device=self.device
            )

            forces = self.object_disturb_force[env_ids].unsqueeze(1)
            torques = self.object_disturb_torque[env_ids].unsqueeze(1)

            self.object.set_external_force_and_torque(
                forces=forces,
                torques=torques,
                env_ids=env_ids,
                body_ids=body_ids,
                is_global=True,
            )
    # ------------------------------------------------------------------ #
    # Gait success test (HORA logic)
    # ------------------------------------------------------------------ #
    def _check_gait_success(self):
        obj_pos = self.object.data.root_pos_w

        tip_pos = self.hand.data.body_pos_w.index_select(
            1, self.finger_bodies
        )

        dist = torch.norm(tip_pos - obj_pos[:, None, :], dim=-1)
        cond1 = (dist < float(self.cfg.fingertip_dist_thresh)).all(dim=1)

        tip_force_mag = torch.stack(
            [sensor.data.net_forces_w.norm(dim=-1)
            for sensor in self.fingertip_contact_sensors.values()],
            dim=1,
        )
        in_contact = tip_force_mag > float(self.cfg.min_tip_force)

        cond2 = in_contact.sum(dim=1) >= int(self.cfg.min_contact_fingers)
        cond2 = cond2.squeeze(-1)  # defensive

        cond3 = obj_pos[:, 2] > float(self.cfg.reset_z_threshold)

        success = cond1 & cond2 & cond3
        assert success.shape == (self.num_envs,), success.shape
        return success
                # tactile_flow = torch.clamp(tactile_flow, -2.0, 2.0)

    def _get_contact_obs(self):

        pos_xy_list = []
        force_mag_list = []
        contact_flag_list = []

        threshold = float(self.cfg.min_tip_force)

        for sensor in self.fingertip_contact_sensors.values():

            # world-frame data
            contact_pos_w = sensor.data.contact_pos_w        # (N, 3)
            sensor_pos_w = sensor.data.pos_w                 # (N, 3)
            sensor_quat_w = sensor.data.quat_w               # (N, 4)
            force_mag = sensor.data.net_forces_w             # (N, 1)  <-- magnitude
            # print("contact_pos_w", contact_pos_w)
            # print("sensor_pos_w", sensor_pos_w)
            # print("force_mag", force_mag)

            # ----------- HANDLE RESET CASE -----------
            # ---- squeeze singleton dims ----
            contact_pos_w = sensor.data.contact_pos_w.squeeze(1).squeeze(1)  # (N,3)
            sensor_pos_w  = sensor.data.pos_w.squeeze(1)                     # (N,3)
            sensor_quat_w = sensor.data.quat_w.squeeze(1)                    # (N,4)

            # ----- force magnitude -----
            force_vec = sensor.data.net_forces_w.squeeze(1)                  # (N,3)
            force_mag = torch.norm(force_vec, dim=-1, keepdim=True)          # (N,1)

            contact_flag = (force_mag > threshold).float()

            # ----- position -----
            pos_xy = torch.zeros(contact_pos_w.shape[0], 2, device=self.device)

            active = contact_flag.squeeze(-1) > 0

            if active.any():
                cp = torch.nan_to_num(contact_pos_w[active], nan=0.0)
                sp = sensor_pos_w[active]
                sq = sensor_quat_w[active]

                pos_rel_w = cp - sp

                pos_rel_local = quat_apply_inverse(sq, pos_rel_w)

                pos_xy[active] = pos_rel_local[:, :2]

            # ----- mask force -----
            force_mag = force_mag * contact_flag

            # ---- append ----
            pos_xy_list.append(pos_xy)
            force_mag_list.append(force_mag)
            contact_flag_list.append(contact_flag)

        # ---- concatenate across fingertips ----
        pos_xy_all = torch.cat(pos_xy_list, dim=-1)          # (N, 2*num_tips)
        force_all = torch.cat(force_mag_list, dim=-1)        # (N, num_tips)
        flag_all = torch.cat(contact_flag_list, dim=-1)      # (N, num_tips)
        # print("pos_xy_all", pos_xy_all)
        # print("force_all", force_all)
        # print("flag_all", flag_all)
        contact_obs = torch.cat([pos_xy_all, force_all, flag_all], dim=-1)
        return contact_obs

    def _get_observations(self):

        if self.cfg.if_tactile_vis:
            for idx, ts in self.tactile_sensors.items():
                if not self._if_loaded_tactile_reference_image[idx]:
                    ts._load_tactile_reference_images()
                    self._if_loaded_tactile_reference_image[idx] = True
                _, _ = ts._get_tactile_images_tensors()   # [B, C, H, W], for rsl-rl

        if self.obs_type == "tactile":
            depth_maps = []
            tactile_flows = []
            for idx, ts in self.tactile_sensors.items():
                # save reference image (if enabled)
                if self.cfg.tactile_sensor_cfgs[idx].if_save_tactile_reference_image:
                    ts._save_tactile_reference_images()
                # load reference once
                elif not self._if_loaded_tactile_reference_image[idx]:
                    ts._load_tactile_reference_images()
                    self._if_loaded_tactile_reference_image[idx] = True
                depth_map, tactile_flow = ts._get_tactile_images_tensors()   # [B, C, H, W], for rsl-rl
                depth_map = torch.clamp(depth_map, 0.0, 1.0)
                depth_maps.append(depth_map)
                tactile_flows.append(tactile_flow)
            depth_maps_obs = torch.cat(depth_maps, dim=2)  # width dimension
            tactile_flows_obs = torch.cat(tactile_flows, dim=2)  # width dimension
            
            prev_depth = self.depth_map_lag_history[:, 1:].clone()
            prev_flow = self.tactile_flow_lag_history[:, 1:].clone()

            cur_depth = depth_maps_obs.unsqueeze(1)
            cur_flow = tactile_flows_obs.unsqueeze(1)

            # TODO: do we need large image buffer?
            self.depth_map_lag_history = torch.cat([prev_depth, cur_depth], dim=1)
            self.tactile_flow_lag_history = torch.cat([prev_flow, cur_flow], dim=1)
            long_depth_map = self.depth_map_lag_history[:, -self.cfg.obs_hist_len:].clone()
            long_tactile_flow = self.tactile_flow_lag_history[:, -self.cfg.obs_hist_len:].clone()

        # --------------------------------------------------
        # Proprioceptive observation (o_t)
        # --------------------------------------------------
        prev_proprio_obs_buf = self.proprio_buf_lag_history[:, 1:].clone()

        joint_noise_matrix = (
            (torch.rand(self.hand_dof_cur_pos.shape, device=self.device) * 2.0 - 1.0)
            * self.cfg.joint_noise_scale
        )

        cur_joint_pos = unscale(
            joint_noise_matrix + self.hand_dof_cur_pos,
            self.hand_dof_lower_limits,
            self.hand_dof_upper_limits,
        ).clone().unsqueeze(1)

        cur_tar_pos = self.cur_targets[:, None]
        cur_proprio_obs_buf = torch.cat([cur_joint_pos, cur_tar_pos], dim=-1)

        self.proprio_buf_lag_history[:] = torch.cat([prev_proprio_obs_buf, cur_proprio_obs_buf], dim=1)

        # flatten last 3 frames → o_t ∈ R^96
        short_proprio_obs = (
            self.proprio_buf_lag_history[:, -3:]
            .reshape(self.num_envs, -1)
            .clone()
        )

        long_proprio_obs = self.proprio_buf_lag_history[:, -self.cfg.obs_hist_len:].clone()

        if self.cfg.if_use_low_dim_contact_obs:
            cur_contact_obs = self._get_contact_obs()  # (N, contact_dim)
            # shift history
            self.contact_buf_lag_history = torch.roll(
                self.contact_buf_lag_history,
                shifts=-1,
                dims=1,
            )
            # insert newest frame
            self.contact_buf_lag_history[:, -1] = cur_contact_obs
            short_contact_obs = (
                self.contact_buf_lag_history[:, -3:]
                .reshape(self.num_envs, -1)
            )
            long_contact_obs = self.contact_buf_lag_history[:, -self.cfg.obs_hist_len:].clone()

            cur_proprio = cur_proprio_obs_buf.squeeze(1)     # (N, proprio_dim)
            cur_contact = cur_contact_obs                   # (N, contact_dim)

            cur_proprio_contact = torch.cat(
                [cur_proprio, cur_contact],
                dim=-1
            ).unsqueeze(1)                                   # (N,1,total_dim)
            prev_buf = self.proprio_and_contact_buf_lag_history[:, 1:].clone()

            self.proprio_and_contact_buf_lag_history[:] = torch.cat(
                [prev_buf, cur_proprio_contact],
                dim=1,
            )
            long_proprio_and_contact_obs = (
                self.proprio_and_contact_buf_lag_history[
                    :, -self.cfg.obs_hist_len:
                ].clone()
            )
        # --------------------------------------------------
        # Privileged observation (e_t)
        # --------------------------------------------------
        priv_vec = []
        if not hasattr(self, "_scale_initialized"):
            self._retrieve_object_scale_from_usd()
            self._scale_initialized = True

        if hasattr(self, "object_position_sceneframe"):
            if self.cfg.if_use_positional_encoding:
                priv_vec.append(positional_encoding(self.object_position_sceneframe.clone()))  # (N, 3)
            else:
                priv_vec.append(self.object_position_sceneframe)          # (N, 3)
        if hasattr(self, "object_rotate_velocity"):
            priv_vec.append(self.object_rotate_velocity)
            # print("object_rotate_velocity", self.object_rotate_velocity)
        if hasattr(self.hand.data, "root_quat_w"):
            # hand_rpy = rpy_from_quat(self.hand.data.root_quat_w)
            # priv_vec.append(hand_rpy)
            priv_vec.append(self.hand.data.root_quat_w.clone())          # (N, 4)

        if not self.cfg.priv_obs_norm:
            if hasattr(self, "object_mass"):
                priv_vec.append(self.object_mass)         # (N, 1)
            if hasattr(self, "object_com"):
                priv_vec.append(self.object_com)          # (N, 3)
            if hasattr(self, "object_static_friction"):
                priv_vec.append(self.object_static_friction)      # (N, 1)
            if hasattr(self, "object_dynamic_friction"):
                priv_vec.append(self.object_dynamic_friction)     # (N, 1)
            if hasattr(self, "object_restitution"):
                priv_vec.append(self.object_restitution)          # (N, 1)
            if hasattr(self, "object_scale"):
                priv_vec.append(self.object_scale)
            if hasattr(self, "object_disturb_force") and self.cfg.if_force_disturbance:
                priv_vec.append(self.object_disturb_force)
                print("object_disturb_force", self.object_disturb_force)
            if hasattr(self, "object_disturb_torque") and self.cfg.if_force_disturbance:
                priv_vec.append(self.object_disturb_torque)
                # print("object_disturb_torque", self.object_disturb_torque)

        else:
            if hasattr(self, "object_scale"):
                scale_min, scale_max = 0.9, 1.3  # 0.975, 1.025
                scale_norm = unscale(
                    self.object_scale,
                    torch.tensor(scale_min, device=self.device),
                    torch.tensor(scale_max, device=self.device),
                )
                priv_vec.append(scale_norm)

            if hasattr(self, "object_mass"):
                mass_min, mass_max = 0.01, 0.25
                mass_norm = unscale(
                    self.object_mass,
                    torch.tensor(mass_min, device=self.device),
                    torch.tensor(mass_max, device=self.device),
                )
                priv_vec.append(mass_norm)

            if hasattr(self, "object_com"):
                com_min = torch.tensor([-0.015, -0.015, -0.015], device=self.device)
                com_max = torch.tensor([ 0.015,  0.015,  0.015], device=self.device)
                com_norm = unscale(
                    self.object_com,
                    com_min,
                    com_max,
                )
                priv_vec.append(com_norm)

            if hasattr(self, "object_static_friction"):
                fric_min, fric_max = 0.3, 3.0
                static_fric_norm = unscale(
                    self.object_static_friction,
                    torch.tensor(fric_min, device=self.device),
                    torch.tensor(fric_max, device=self.device),
                )
                priv_vec.append(static_fric_norm)

            if hasattr(self, "object_dynamic_friction"):
                fric_min, fric_max = 0.3, 3.0
                dynamic_fric_norm = unscale(
                    self.object_dynamic_friction,
                    torch.tensor(fric_min, device=self.device),
                    torch.tensor(fric_max, device=self.device),
                )
                priv_vec.append(dynamic_fric_norm)

            if hasattr(self, "object_restitution"):
                rest_min, rest_max = 0.0, 0.2
                restitution_norm = unscale(
                    self.object_restitution,
                    torch.tensor(rest_min, device=self.device),
                    torch.tensor(rest_max, device=self.device),
                )
                priv_vec.append(restitution_norm)
            if hasattr(self, "object_disturb_force") and self.cfg.if_force_disturbance:
                force_min, force_max = self.cfg.force_range
                force_norm = unscale(
                    self.object_disturb_force,
                    torch.tensor(force_min, device=self.device),
                    torch.tensor(force_max, device=self.device),
                )
                priv_vec.append(force_norm)
                # print("normed object_disturb_force", force_norm)
            if hasattr(self, "object_disturb_torque") and self.cfg.if_force_disturbance:
                torque_min, torque_max = self.cfg.torque_range
                torque_norm = unscale(
                    self.object_disturb_torque,
                    torch.tensor(torque_min, device=self.device),
                    torch.tensor(torque_max, device=self.device),
                )
                priv_vec.append(torque_norm)

        priv_obs = torch.cat(priv_vec, dim=-1)        # (N, priv_dim)
        # print("priv_obs", priv_obs)

        obs = {
            "short_proprio": short_proprio_obs,
            "long_proprio": long_proprio_obs,
            "priv": priv_obs,
        }
        if self.cfg.if_use_low_dim_contact_obs:
            obs.update({
                "short_contact": short_contact_obs,
                "long_contact": long_contact_obs,
                "long_proprio_and_contact_obs": long_proprio_and_contact_obs,
            })
        if self.obs_type == "tactile":
            obs.update({
                "tactile_flow": tactile_flows_obs,
                "depth_map": depth_maps_obs,
                "long_tactile_flow": long_tactile_flow,
                "long_depth_map": long_depth_map,
            })
        if self.obs_type in ("oracle", "tactile"):
            return obs
        else:
            raise NotImplementedError(f"Unsupported obs_type: {self.obs_type}")

    def apply_random_force_to_object(
        self,
        env_ids: torch.Tensor,
        force_range: tuple = (-5.0, 5.0),   # (min, max)
        torque_range: tuple = (-1.0, 1.0),
        axes=("x", "y", "z"),
    ):
        num_envs = len(env_ids)
        body_ids = torch.arange(self.object.num_bodies, device=self.device)
        num_bodies = len(body_ids)

        forces = torch.zeros(num_envs, num_bodies, 3, device=self.device)
        torques = torch.zeros(num_envs, num_bodies, 3, device=self.device)

        axis_map = {"x": 0, "y": 1, "z": 2}

        for ax in axes:
            forces[:, :, axis_map[ax]] = torch.empty(
                num_envs, num_bodies, device=self.device
            ).uniform_(*force_range)

        torques.uniform_(*torque_range)

        self.object.set_external_force_and_torque(
            forces=forces,
            torques=torques,
            env_ids=env_ids,
            body_ids=body_ids,
            is_global=True,
        )

    def _get_rewards(self):
        self.rot_axis_buf[:, -1] = -1
        # --- quaternion delta ---
        quat_diff = quat_mul(
            self.object_rot,
            quat_conjugate(self.object_rot_prev),
        )
        # --- axis-angle representation ---
        axis_angle = axis_angle_from_quat(quat_diff)   # (N, 3)
        # this already equals axis * angle  (radians)
        # --- convert to angular velocity ---
        dt_effective = self.cfg.sim.dt * self.cfg.decimation
        object_angvel = axis_angle / dt_effective     # rad/s
        # --- project onto rotation axis ---
        vec_dot = (object_angvel * self.rot_axis_buf).sum(-1)
        obj_rotate_reward = torch.clip(
            vec_dot,
            min=-self.cfg.angvel_clip_range,
            max=self.cfg.angvel_clip_range,
        )
        self.object_rotate_velocity = vec_dot.clone().unsqueeze(-1)

        # obj_linear_vel_penalty = torch.norm(self.object.data.root_lin_vel_w, p=1, dim=-1)
        object_linvel = (
            self.object_position_sceneframe
            - self.object_position_prev_sceneframe
        ) / dt_effective

        obj_linear_vel_penalty = torch.norm(object_linvel, p=1, dim=-1)
        # # Detect drop
        # obj_height = self.object_position_sceneframe[:, 2]
        # dropped = obj_height < self.cfg.reset_z_threshold

        # # Large drop penalty
        # drop_penalty = torch.zeros_like(obj_linear_vel_penalty)
        # drop_penalty[dropped] = -100.0   # choose magnitude

        pose_diff_penalty = (
            (self.hand_dof_cur_pos - self.init_pose_buf) ** 2
        ).sum(-1)

        if self.cfg.ctrl_mode == "position":
            joint_pos = self.hand.data.joint_pos
            joint_vel = self.hand.data.joint_vel
            pos_error = self.cur_targets - joint_pos
            pos_error_penalty = (pos_error ** 2).sum(-1)
            tau_proxy = self.cfg.p_gain * pos_error \
                        - self.cfg.d_gain * joint_vel

            power_proxy = (tau_proxy * joint_vel).sum(-1)
            work_proxy_penalty = power_proxy ** 2

        elif self.cfg.ctrl_mode == "torque":

            # pos_error_penalty = (self.torques ** 2).sum(-1)
            # work_proxy_penalty = (
            #     (self.torques * self.hand_dof_cur_vel).sum(-1)
            # ) ** 2
            pos_error_penalty = (self.torques ** 2).sum(-1)
            # ---- finite difference joint velocity ----
            dof_vel_finite_diff = (
                (self.hand_dof_cur_pos - self.prev_hand_dof_pos)
                / dt_effective
            )
            # ---- work penalty (IDENTICAL to original) ----
            work_proxy_penalty = (
                (self.torques * dof_vel_finite_diff).sum(-1)
            ) ** 2

        # print("pos_error_penalty", pos_error_penalty)
        # print("work_proxy_penalty", work_proxy_penalty)
        # print("self.object_angvel", self.object_angvel)
        # print("obj_linear_vel_penalty", obj_linear_vel_penalty)
        # print("pose_diff_penalty", pose_diff_penalty.mean().item())
        # print("power_proxy", pos_error_penalty.mean().item())
        # print("work_proxy_penalty", work_proxy_penalty.mean().item())
        # Return a zero reward for all envs.
        return self._compute_rewards(
            obj_rotate_reward=obj_rotate_reward,
            obj_rotate_reward_scale=self.cfg.obj_rotate_reward_scale,
            obj_linear_vel_penalty=obj_linear_vel_penalty,
            obj_linear_vel_penalty_scale=self.cfg.obj_linear_vel_penalty_scale,
            pose_diff_penalty=pose_diff_penalty,
            pose_diff_penalty_scale=self.cfg.pose_diff_penalty_scale,
            # drop_penalty=drop_penalty,
            pos_error_penalty=pos_error_penalty,
            pos_error_penalty_scale=self.cfg.pos_error_penalty_scale,
            work_penalty=work_proxy_penalty,
            work_penalty_scale=self.cfg.work_penalty_scale,
        )

    def _compute_rewards(
        self,
        obj_rotate_reward: torch.Tensor,
        obj_rotate_reward_scale: float,
        obj_linear_vel_penalty: torch.Tensor,
        obj_linear_vel_penalty_scale: float,
        pose_diff_penalty: torch.Tensor,
        pose_diff_penalty_scale: float,
        # drop_penalty: torch.Tensor,
        pos_error_penalty: torch.Tensor,
        pos_error_penalty_scale: float,
        work_penalty: torch.Tensor,
        work_penalty_scale: float,
    ):
        
        obj_rot_r = obj_rotate_reward * obj_rotate_reward_scale
        obj_lin_p = obj_linear_vel_penalty * obj_linear_vel_penalty_scale
        pos_p = pose_diff_penalty * pose_diff_penalty_scale
        pos_error_p = pos_error_penalty * pos_error_penalty_scale
        work_p = work_penalty * work_penalty_scale

        rewards = obj_rot_r + obj_lin_p + pos_p + pos_error_p + work_p 

        # rewards += drop_penalty
        self.extras["log"] = {
            "obj_rot_r": obj_rot_r.mean(),
            "obj_lin_p": obj_lin_p.mean(),
            "pos_p": pos_p.mean(),
            "pos_error_p": pos_error_p.mean(),
            "work_p": work_p.mean(),
            # "drop_penalty": drop_penalty.mean(),
            "reward": rewards.mean(),
        }
        return rewards
    
    def _compute_intermediate_values(self):

        if self.cfg.if_frames_vis:
            print("Testing visualizer")
            self.obj_marker.visualize(self.object.data.root_pos_w, self.object.data.root_quat_w)
            self.hand_marker.visualize(self.hand.data.body_pos_w[:, 0], self.hand.data.body_quat_w[:, 0])
        if self.cfg.if_testing:
            # print("Mass:", self.object_mass)
            # print("Static friction:", self.object_static_friction)
            # print("Dynamic friction:", self.object_dynamic_friction)
            # print("Restitution:", self.object_restitution)
            # print("COM:", self.object_com)
            # print("object scale:", self.object_scale)
            print("Testing shorter episode.")
        self.prev_hand_dof_pos = self.hand_dof_cur_pos.clone()
        self.hand_dof_cur_pos = self.hand.data.joint_pos
        self.hand_dof_cur_vel = self.hand.data.joint_vel

        # object
        self.object_position_prev_sceneframe = self.object_position_sceneframe.clone()

        self.object_position_sceneframe = (self.object.data.root_pos_w - self.scene.env_origins).clone()
        self.object_rot_prev = self.object_rot.clone()
        self.object_rot = self.object.data.root_pose_w[:, 3:7]

    def _get_dones(self):
        # No termination or truncation logic for now
        self._compute_intermediate_values()
        # obj_height = self.object_position_sceneframe[:, 2]
        # terminated_obj = obj_height < self.cfg.reset_z_threshold
        obj_hand_anchor_dist = torch.norm(self.object_position_sceneframe - self.object_anchor_pos_sceneframe, dim=-1)
        terminated_obj = obj_hand_anchor_dist > self.cfg.reset_anchor_dist_threshold
        terminated = terminated_obj
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _pre_physics_step(self, actions: torch.Tensor) -> None:

        if self.cfg.ctrl_mode == "torque":
            self.actions = torch.clamp(actions, -self.cfg.action_clip, self.cfg.action_clip)
            targets = self.prev_targets + (1.0 / 24.0) * self.actions
            self.cur_targets = saturate(
                targets,
                self.hand_dof_lower_limits,
                self.hand_dof_upper_limits,
            )
            self.prev_targets[:] = self.cur_targets.clone()

        elif self.cfg.ctrl_mode == "position":
            self.actions = actions.clone().to(self.device) * self.cfg.actions_scale
            # env_ids = torch.arange(self.num_envs, device=self.device)
        if self.cfg.if_force_disturbance:
            self.object.write_data_to_sim()

    def _retrieve_object_scale_from_usd(self):

        stage = get_context().get_stage()

        prim_paths = sim_utils.find_matching_prim_paths(
            self.cfg.object_cfg.prim_path
        )

        scales = torch.ones((self.num_envs, 1), device=self.device)

        for i, prim_path in enumerate(prim_paths):
            prim = stage.GetPrimAtPath(prim_path)

            # check root first
            xform = UsdGeom.Xformable(prim)
            found = False

            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale_vec = op.Get()
                    scales[i, 0] = float(scale_vec[0])
                    found = True
                    break

            # if not found, check children
            if not found:
                for child in prim.GetChildren():
                    child_xform = UsdGeom.Xformable(child)
                    for op in child_xform.GetOrderedXformOps():
                        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                            scale_vec = op.Get()
                            scales[i, 0] = float(scale_vec[0])
                            found = True
                            break
                    if found:
                        break

        self.object_scale = scales
        
    def _update_object_physics_properties(self, env_ids: torch.Tensor):
        """
        Retrieve randomized object physics parameters after EventTerm execution.
        """

        # MASS (CPU tensor)
        # shape: (num_envs, num_bodies)
        masses = self.object.root_physx_view.get_masses()
        env_ids_cpu = env_ids.cpu()
        self.object_mass[env_ids, 0] = masses[env_ids_cpu, 0].to(self.device)

        # MATERIAL (CPU tensor)
        materials = self.object.root_physx_view.get_material_properties()
        # shape: (N_env, N_shape, 3)
        # order: [static, dynamic, restitution]
        selected_material = materials[env_ids_cpu, 0]  # (len(env_ids), 3)
        self.object_static_friction[env_ids, 0] = selected_material[:, 0].to(self.device)
        self.object_dynamic_friction[env_ids, 0] = selected_material[:, 1].to(self.device)
        self.object_restitution[env_ids, 0] = selected_material[:, 2].to(self.device)

        # COM (CPU tensor)
        # shape: (num_envs, num_bodies, 3)
        com = self.object.root_physx_view.get_coms()
        # assuming single rigid body object
        self.object_com[env_ids] = com[env_ids_cpu, 0:3].to(self.device)

    def _apply_action(self) -> None:
        # scale actions to SMALL increments
        # print("testing, action set to None")
        # pass
        if self.cfg.ctrl_mode == "torque":
            joint_pos = self.hand.data.joint_pos
            joint_vel = self.hand.data.joint_vel

            self.torques = torch.clamp(
                self.cfg.p_gain * (self.cur_targets - joint_pos)
                - self.cfg.d_gain * joint_vel,
                -self.cfg.torque_limit,
                self.cfg.torque_limit,
            )

            self.hand.set_joint_effort_target(self.torques)
            
        elif self.cfg.ctrl_mode == "position":
            delta = self.actions * self.cfg.max_position_delta
            # moving average for smoother action (on increments)
            delta = (
                self.cfg.act_moving_average * delta
                + (1.0 - self.cfg.act_moving_average)
                * (self.prev_targets - self.cur_targets)
            )
            ###################joint impulse perturbation##########################
            num_envs, num_dofs = delta.shape
            device = delta.device
            # --------------------------------------------------
            # 1. Start impulse only if not already active
            # --------------------------------------------------
            if self.cfg.if_random_joint_impulse:
                start_mask = (
                    (torch.rand(num_envs, device=device) < self._impulse_prob)
                    & (self._impulse_timer == 0)
                )
                if start_mask.any():
                    env_ids = torch.nonzero(start_mask, as_tuple=False).squeeze(-1)

                    # Fix one joint for the whole duration
                    self._impulse_joint[env_ids] = torch.randint(
                        0, num_dofs, (env_ids.shape[0],), device=device
                    )
                    self._impulse_timer[env_ids] = self._impulse_duration
                    # print(f"Start consecutive impulse in envs: {env_ids.tolist()}")
                # --------------------------------------------------
                # 2. Apply impulse while active (same joint)
                # --------------------------------------------------
                active_mask = self._impulse_timer > 0
                if active_mask.any():
                    env_ids = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
                    # Signed constant impulse
                    impulse = torch.sign(torch.randn(env_ids.shape[0], device=device)) \
                            * self._impulse_strength
                    delta[env_ids, self._impulse_joint[env_ids]] += impulse
                    self._impulse_timer[env_ids] -= 1
            ############################################################

            # incremental update
            self.cur_targets += delta

            # clip to joint limits
            self.cur_targets = saturate(
                self.cur_targets,
                self.hand_dof_lower_limits,
                self.hand_dof_upper_limits,
            )

            # update prev_targets
            self.prev_targets = self.cur_targets.clone()

            # apply target
            self.hand.set_joint_position_target(self.cur_targets)

        
@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention
