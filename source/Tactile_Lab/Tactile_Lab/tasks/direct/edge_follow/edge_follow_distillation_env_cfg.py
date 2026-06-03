# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG, UR5_TACTIP_CFG   # isort:skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns, TiledCameraCfg
import torch
from gymnasium import spaces
from importlib import resources
from pathlib import Path


def asset(path: str) -> str:
    return str(
        Path(resources.files("Tactile_Lab.tactile_lab_assets")) / path
    )

@configclass
class EdgeDistillationEnvCfg(DirectRLEnvCfg):
    # env
    # obs_type = "oracle"  # tactile, oracle
    obs_type = "tactile"  # tactile, oracle
    if obs_type == "oracle":
        num_envs = 16384  # for oracle
        edge_stop_distance = None  # if None, no stop distance check, 0.01. 
        success_threshold = 0.01  # success threshold to goal for oracle
    else:
        num_envs = 2048  # for tactile img, 32x32, 2048
        edge_stop_distance = 0.01  # if None, no stop distance check, 0.01
        success_threshold = 0.015  # success threshold to goal for tactile

    episode_length_s = 6  # in second, which equals to 500 timesteps = episode_length_s/(decimation * sim_dt) 8.3333
    decimation = 2
    action_space = 2  # 2D pose control (xy)
    tactile_img_size = 32

    if_use_positional_encoding = True  # whether to use positional encoding for oracle observation
    if obs_type == "oracle":
        if if_use_positional_encoding:
            observation_space = 56  # positional encoding: 2*8 + 2*8 + 2*8 + 8 = 56
        else:
            observation_space = 7  
    elif obs_type == "tactile":
        # observation_space = (tactile_img_size, tactile_img_size, 1)  # tactile image flattened
        observation_space = spaces.Box(
                low=float("-inf"), high=float("inf"), shape=(tactile_img_size, tactile_img_size, 1)
        )
    state_space = 0
    # robot_name = "franka_panda"
    # robot_name = "ur10"
    robot_name = "ur5_tactip"
    if_debug = False

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

    # tactile sensor
    tactile_image_type = 'depth'  # 'rgb' or 'depth' or 'depth_original' or 'depth_and_shear'
    if "depth" in tactile_image_type:
        if_depth = True
    if_shear_arrows = False
    if_shear_hsv = False

    if_render_tactile = False  # only works when obs_type is tactile
    sensor_type = "tactip"
    near_plane = 0.001  # cannot be too small otherwise depth image will be invalid
    far_plane = 1.0
    tactile_img_size = tactile_img_size
    focal_length = 26.5  # in mm
    if_save_tactile_reference_image = False
    tactile_depth_enhance_scale = 1 / 0.0197  # scale the depth difference to enhance the deformation visibility. This is computed by getting the maximun depth of difference (our case is 0.0197m, )
    max_shear_mag = 2.0  # maximum shear magnitude for normalization

    # tactile_camera = CameraCfg(  # Do not use this normal camera class, which cannot be scaled more than 100 envs.
    tactile_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/tcp_link/tactile_cam",
        update_period=0.1,
        height=tactile_img_size,
        width=tactile_img_size,
        update_latest_camera_pose=True,
        data_types=["rgb", "distance_to_image_plane", "motion_vectors"],  # TODO: rgb and motion_vectors are temporary for debugging, should drop it if doing training
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(near_plane, far_plane)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.065), rot=(0, 1, 0, 0), convention="ros"),
    )

    # Goal: non-physical object to mark goal position
    goal_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Goal",
        spawn=sim_utils.MeshSphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # TODO: opacity=0.01 not working
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),  # No collision
            rigid_props=sim_utils.RigidBodyPropertiesCfg(  # Fixed base
                # kinematic_enabled=True,  # Enable kinematic will cause the object to not respond to physics at all, including random pose reset!
                disable_gravity=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.65, 0.0, 0.0)),
    )

    # Edge: rigid object  (custom)
    edge_height = 0.035
    edge_length = 0.18
    if if_save_tactile_reference_image:
        edge_offset_for_save_image = 1.0
    else:
        edge_offset_for_save_image = 0.0
    edge_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Edge",
        spawn=sim_utils.UsdFileCfg(
            usd_path=asset("Robots/tg3_asset/long_edge.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(  # Fixed base
                # kinematic_enabled=True,  # Enable kinematic will cause the object to not respond to physics at all, including random pose reset!
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),  # No collision
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.65 + edge_offset_for_save_image, 0.0, 0.0)),
    )

    # # rigid object (built-in cuboid)
    # edge_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Robot/Edge",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.18, 0.035, 0.035),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(  # Fixed base
    #             # kinematic_enabled=True,   
    #             disable_gravity=True,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),  # No collision
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.8)),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.65, 0.0, 0.035/2)),
    # )

    # robot_cfg
    workframe_scene = [0.65, 0, 0.022, 0.0, 0.0, 0.0]
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
    if obs_type == "oracle":
        robot_ee_speed_scale = [0.05, 0.05, 0.1, 0.1, 0.1, 1.5]  # m/s, rad/s, TODO: finetune the rpy scale
    elif obs_type == "tactile":
        robot_ee_speed_scale = [0.05, 0.05, 0.1, 0.1, 0.1, 1.5]  # m/s, rad/s, TODO: finetune the rpy scale
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

