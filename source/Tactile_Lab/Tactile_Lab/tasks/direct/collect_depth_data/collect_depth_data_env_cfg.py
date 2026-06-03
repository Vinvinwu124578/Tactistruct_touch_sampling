# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from numpy import square
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from Tactile_Lab.tactile_lab_assets.tactile_lab_assets.robots.tg3_ur5 import UR5_TACTIP_CFG   # isort:skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from ipdb import set_trace
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns, TiledCameraCfg
import torch
from gymnasium import spaces
from Tactile_Lab.utility.tactile_sensor import TactileSensorCfg
from Tactile_Lab.utility.utils import rpy_quat_convert
import os

@configclass
class CollectDepthDataEnvCfg(DirectRLEnvCfg):
    obs_type = "tactile"   # "oracle" | "tactile"
    episode_length_s = 6
    decimation = 2
    action_space = 2
    if_use_positional_encoding = True
    tactile_sensor_cfg = None
    tactile_img_size = 256

    num_poses = 100
    tap_or_shear = 'tap'
    shuffle_data = False
    tactile_image_type = 'edge_2d'
    # define dir where real data is stored
    # target_dir = os.path.join(
    #     os.path.dirname(__file__),
    #     f'data/real/{tactile_image_type}/'
    # )
    target_dir = None

    target_home_dir = os.path.join(target_dir, tap_or_shear) if target_dir is not None else None

    target_dir_name = 'csv_val'

    collect_params = {
        "pose_llims": (-5, 0, 3, 0, 0, -180),
        "pose_ulims": (5, 0, 4, 0, 0,  180),
        "sort": True,
        "edge":    (0, 0, 0, 0, 0, 0),
        # "surface": (-50, 0, 0, 0, 0, 0)
        }

    sensor_params = {
        "type": "standard_tactip",
        "image_size": (tactile_img_size, tactile_img_size)
    }
    
    if obs_type == "oracle":
        # ---------------- oracle ----------------
        num_envs = 16384
        edge_stop_distance = None
        success_threshold = 0.01
        if if_use_positional_encoding:
            observation_space = 56
        else:
            observation_space = 7

    elif obs_type == "tactile":
        # ---------------- tactile ----------------
        num_envs = 1
        cam_rpy = [3.14, 3.14, 1.57]  # for using franka_panda_tactip_flip
        cam_quat = rpy_quat_convert(rpy=cam_rpy)  # [w, x, y, z]
        tactile_sensor_cfg = TactileSensorCfg(
            if_render_tactile=True,
            if_save_tactile_reference_image=False,
            tactile_img_size=tactile_img_size,
            offset_pos=(0.0, 0.0, -0.065),  # in meter, translation from tcp frame to tactile sensor frame
            offset_rot=cam_quat,  # in degree, rotation from tcp frame to tactile sensor frame
        )

        observation_space = spaces.Box(
            low=float("-inf"),
            high=float("inf"),
            shape=(
                tactile_sensor_cfg.tactile_img_size,
                tactile_sensor_cfg.tactile_img_size,
                1,
            ),
        )
    else:
        raise ValueError(f"Unknown obs_type: {obs_type}")

    state_space = 0
    # robot_name = "franka_panda"
    # robot_name = "ur10"
    robot_name = "ur5_tactip"
    if_debug = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=num_envs, env_spacing=2.0,  # replicate_physics=True, clone_in_fabric=True
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Edge: rigid object  (custom)
    well_defined_workframe_x = 0.65
    square_height = 0.0275
    square_length = 0.10
    if tactile_sensor_cfg is not None and tactile_sensor_cfg.if_save_tactile_reference_image:
        edge_offset_for_save_image = 1.0
    else:
        edge_offset_for_save_image = 0.0
    edge_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Edge",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/bourne/IsaacLab/source/isaaclab_assets/data/Robots/tg3_asset/square.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(  # Fixed base
                # kinematic_enabled=True,  # Enable kinematic will cause the object to not respond to physics at all, including random pose reset!
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),  # No collision
            # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(well_defined_workframe_x + edge_offset_for_save_image, 0.0, 0.0)),
    )
    
    # robot_cfg
    workframe_scene = [well_defined_workframe_x - square_length/2, 0, square_height, 3.14, 0.0, 0.0]
    tcp_lims_single = [
        [-0.18, 0.18],                          # x lims
        [-0.18, 0.18],                          # y lims
        [-0.0, 0.0],                          # z lims
        [-0.0, 0.0],                          # roll lims
        [-0.0, 0.0],                          # pitch lims
        [-0.0, 0.0]                           # yaw lims
    ]  # or torch.double if needed

    # Expand to (num_env, 6, 2)
    # task-specific joint positions
    # The speed is increased to improve exploration efficiency.
    # However, when using tactile observation, a high speed may cause the tactile sensor to miss contacts and cannot recover due to the lack of global obs. 

    robot_ee_speed_scale = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # m/s, rad/s, TODO: finetune the rpy scale
    # robot_ee_speed_scale = [0.5, 0.5, 0.1, 0.1, 0.1, 1.5]  # m/s, rad/s, TODO: finetune the rpy scale

    new_joint_positions = {
        "base_joint": 0.1688,
        "shoulder_joint": -2.1589,
        "elbow_joint": -1.6435,
        "wrist_1_joint": -0.9086,
        "wrist_2_joint": 1.5717,
        "wrist_3_joint": -2.9538,
    }
    robot_cfg = UR5_TACTIP_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UR5_TACTIP_CFG.spawn.replace(
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False  # disable this does not work for entire arms, only work for the root link
            )
        ),
        init_state=UR5_TACTIP_CFG.init_state.replace(
            joint_pos=new_joint_positions  # replace the init_state inside the config
        ),
    )
    action_scale = 1
    dof_velocity_scale = 0.1

    # reward scales
    if obs_type == "oracle":
        edge_dist_reward_scale = 2
    elif obs_type == "tactile":
        edge_dist_reward_scale = 1
    goal_dist_reward_scale = 1
    action_penalty_scale = 0.005
    action_rate_penalty_scale = 0.01
