# hand_grasp_env.py
import os
import torch
import numpy as np
from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_mul, quat_from_angle_axis, sample_uniform, combine_frame_transforms
from isaaclab.sensors import ContactSensor
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
from Tactile_Lab.tasks.direct.hand_grasp.hand_grasp_env_cfg import HandGraspEnvCfg
import isaaclab.sim as sim_utils
from ipdb import set_trace
from isaaclab.sensors.camera import TiledCamera
from Tactile_Lab.utility.tactile_sensor import TactileSensor
# from Tactile_Lab.utility.utils import randomize_rotation
from isaaclab.utils.math import (
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    sample_uniform,
)

class HandGraspEnv(DirectRLEnv):
    cfg: HandGraspEnvCfg

    def __init__(self, cfg: HandGraspEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # DOFs
        self.num_hand_dofs = self.hand.num_joints

        # Fingertips
        self.finger_bodies = torch.tensor(
            [self.hand.body_names.index(n) for n in cfg.fingertip_body_names],
            device=self.device,
            dtype=torch.long,
        )

        # Joint limits
        limits = self.hand.root_physx_view.get_dof_limits()
        self.hand_dof_lower = limits[..., 0].to(self.device)
        self.hand_dof_upper = limits[..., 1].to(self.device)

        # Canonical Allegro grasp pose (IsaacLab DOF order!)
        self.canonical_pose = torch.tensor(
            [
                0.082, 1.244, 0.265, 0.298,
                1.104, 1.163, 0.953, -0.138,
                0.005, 1.096, 0.080, 0.150,
                0.029, 1.337, 0.285, 0.317,
            ],
            device=self.device,
        )
        self.object_pose_in_handframe = (
            torch.tensor(self.cfg.object_pose_in_handframe, device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        # Buffers
        self.grasp_step_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        # buffers for position targets
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # Cache
        self.saved_grasps = []

        # Unit axes
        self.x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        self.y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        self.z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)

        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # Debug visualization
        if self.cfg.if_debug:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
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
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        
        spawn_ground_plane("/World/ground", GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["hand"] = self.hand
        self.scene.rigid_objects["object"] = self.object

        if self.cfg.use_tactile:
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

    def randomize_rotation(
        self,
        pose: torch.Tensor,
        env_ids: torch.Tensor,
        axes=("x", "y"),
        ranges=None,
    ):
        """Randomize object rotation around selected axes.

        Args:
            pose: (N, 7) or (N, >=7) pose tensor [pos(3), quat(4), ...]
            env_ids: environment indices
            axes: tuple of axes to randomize, e.g. ("x",), ("z",), ("x","y","z")
            ranges: dict like {"x": pi/4, "y": (-pi/6, pi/6), "z": pi}
        """
        axis_map = {
            "x": self.x_axis[env_ids],
            "y": self.y_axis[env_ids],
            "z": self.z_axis[env_ids],
        }

        # identity quaternion
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)

        for ax in axes:
            if ranges is None or ax not in ranges:
                lo, hi = -np.pi, np.pi
            else:
                r = ranges[ax]
                lo, hi = (-r, r) if isinstance(r, (int, float)) else r

            angle = sample_uniform(lo, hi, (len(env_ids),), device=self.device)
            q = quat_mul(
                quat_from_angle_axis(angle, axis_map[ax]),
                q,
            )

        pose[:, 3:7] = q

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # Randomize hand joint around default open pose
        pos = self.hand._data.default_joint_pos[env_ids] + 0.25 * sample_uniform(
            -1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device
        )
        
        # print("testing default_joint_pos")
        # pos = self.hand._data.default_joint_pos[env_ids]

        pos = torch.clamp(pos, self.hand_dof_lower[0], self.hand_dof_upper[0])
        self.cur_targets[env_ids] = pos
        self.hand.set_joint_position_target(pos, env_ids=env_ids)
        hand_state = self.hand.data.default_root_state[env_ids].clone()
        hand_state[:, :3] += self.scene.env_origins[env_ids]
        # For upside-down grasp, also randomize hand base rotation to be roughly downwards facing, with some noise.
        # print("testing no random hand pose")
        # if self.cfg.task_type == "upsidedown_grasp" or self.cfg.task_type == "upsidedown_no_pins_grasp" :
        if self.cfg.if_random_hand_pose:
            self.randomize_rotation(
                hand_state,
                env_ids,
                axes=("x", "y"),
                ranges={
                    "x": self.cfg.random_hand_rp_range,
                    "y": self.cfg.random_hand_rp_range,
                    "z": self.cfg.random_hand_rp_range,
                },
            )
            hand_state[:, 7:] = 0.0
            self.hand.write_root_pose_to_sim(hand_state[:, :7], env_ids)
            self.hand.write_root_velocity_to_sim(hand_state[:, 7:], env_ids)

        self.hand.write_joint_state_to_sim(pos, torch.zeros_like(pos), env_ids=env_ids)

        self.grasp_step_counter[env_ids] = 0

        # Randomize object pose
        obj_state = self.object.data.default_root_state[env_ids].clone()
        self.randomize_rotation(obj_state, env_ids, axes=("x", "y"))
        obj_pos_in_sceneframe, _ = combine_frame_transforms(
            hand_state[:, :3],  # the hand does not need env_id indexing since it already includes when derived it from default_root_state
            hand_state[:, 3:7],
            self.object_pose_in_handframe[env_ids, :3],
            self.object_pose_in_handframe[env_ids, 3:7],
        )
        obj_state[:, 7:] = 0.0
        obj_state[:, :3] = obj_pos_in_sceneframe[:, :3]  # don't need to add env origin since hand_state already includes it
        # print("testing obj pos remote set")
        # obj_state[:, 0] += 10  #  test remote set
        # obj_state[:, :3] += self.scene.env_origins[env_ids]
        # print("object pos:", obj_state[:, :3])
        self.object.write_root_pose_to_sim(obj_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(obj_state[:, 7:], env_ids)

    # ------------------------------------------------------------------ #
    # Grasp success test (HORA logic)
    # ------------------------------------------------------------------ #
    def _check_grasp_success(self):
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

    def _get_observations(self):
        # Minimal dummy observation
        # Shape must be (num_envs, observation_space)
        if self.cfg.use_tactile:
            tactile_imgs = []
            for idx, ts in self.tactile_sensors.items():
                # save reference image (if enabled)
                if self.cfg.tactile_sensor_cfgs[idx].if_save_tactile_reference_image:
                    ts._save_tactile_reference_images()
                # load reference once
                elif not self._if_loaded_tactile_reference_image[idx]:
                    ts._load_tactile_reference_images()
                    self._if_loaded_tactile_reference_image[idx] = True
                tactile_imgs.append(ts._get_tactile_images_tensors())  # [B,H,W,C]
            # concatenate along width (multi-finger panorama style)
            tactile_obs = torch.cat(tactile_imgs, dim=2)  # width dimension

        obs = torch.zeros(
            (self.num_envs, self.cfg.observation_space),
            device=self.device,
            dtype=torch.float,
        )
        return {"policy": obs}

    def _get_rewards(self):
        # Grasp generation does not use rewards.
        # Return a zero reward for all envs.
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    def _compute_intermediate_values(self):
        if self.cfg.if_testing:
            print("Testing shorter episode.")
        if self.cfg.if_debug:
            self.obj_marker.visualize(self.object.data.root_pos_w, self.object.data.root_quat_w)
            self.hand_marker.visualize(self.hand.data.body_pos_w[:, 0], self.hand.data.body_quat_w[:, 0])
            # set_trace()
            # self.goal_marker.visualize(
            #     self.current_goal_pose_worldframe_all_envs[:, 0:3],
            #     self.current_goal_quat_worldframe_all_envs,
            # )

    def _get_dones(self):
        self._compute_intermediate_values()
        self.grasp_step_counter += 1

        success = self._check_grasp_success()
        assert success.shape == (self.num_envs,), success.shape

        ready = self.grasp_step_counter >= self.cfg.grasp_settle_steps
        if ready.any():
            self._save_grasps(success & ready)

        terminated = ready | (~success)
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        if self.grasp_step_counter[0] % 50 == 0:
            print(f"[DEBUG] Cached grasps: {sum(g.shape[0] for g in self.saved_grasps)}")
        return terminated, truncated
    #     print("testing no termination")
    #     self._compute_intermediate_values()
    #     terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    #     truncated = self.episode_length_buf >= self.max_episode_length - 1
    #     return terminated, truncated

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # print("testing _apply_action")
        # pass
        self.hand.set_joint_position_target(self.cur_targets)
    # ------------------------------------------------------------------ #
    # Save grasps
    # ------------------------------------------------------------------ #
    def _save_grasps(self, success_mask):
        if not success_mask.any():
            return

        hand_pos = self.hand.data.joint_pos[success_mask]

        # Object pose relative to env origin (local frame)
        obj_pose_w = self.object.data.root_state_w[success_mask, :7]
        env_origins = self.scene.env_origins[success_mask]
        obj_pos_local = obj_pose_w[:, :3] - env_origins
        obj_quat_local = obj_pose_w[:, 3:7]
        obj_pose_local = torch.cat([obj_pos_local, obj_quat_local], dim=-1)

        # --- Hand root pose (env-local) ---
        hand_pose_w = self.hand.data.root_state_w[success_mask, :7]
        hand_pos_local = hand_pose_w[:, :3] - env_origins
        hand_quat_local = hand_pose_w[:, 3:7]
        hand_root_pose_local = torch.cat([hand_pos_local, hand_quat_local], dim=-1)
        grasp = torch.cat([hand_pos, hand_root_pose_local, obj_pose_local], dim=-1).cpu()
        self.saved_grasps.append(grasp)

        num_saved = sum(g.shape[0] for g in self.saved_grasps)
        print(f"[INFO] Saved grasps so far: {num_saved}")

        if num_saved >= self.cfg.total_collect_grasp_num:
            save_path = self.cfg.grasp_cache_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            data = torch.cat(self.saved_grasps, dim=0).numpy()
            np.save(save_path, data)
            print(f"[DONE] Saved {data.shape[0]} grasps → {save_path}")
            exit()


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
