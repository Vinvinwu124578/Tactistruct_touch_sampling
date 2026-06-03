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
from Tactile_Lab.tactile_lab_assets.tactile_lab_assets.robots.tg3_ur5 import UR5_RA_TACTIP_CFG   # isort:skip

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from ipdb import set_trace
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns, TiledCameraCfg
import torch
from gymnasium import spaces
from isaaclab.assets.articulation import ArticulationCfg
from Tactile_Lab.utility.tactile_sensor import TactileSensorCfg

@configclass
class ObjPushEnvCfg(DirectRLEnvCfg):
    # env
    obs_type = "oracle"   # "oracle" | "tactile"
    if_use_positional_encoding = False
    # --------------------------------------------------
    # General env settings
    # --------------------------------------------------
    episode_length_s = 18
    decimation = 2
    action_space = 2          # 2D pose control (xy)
    clip_actions = 1.0
    state_space = 0
    # --------------------------------------------------
    # Observation-dependent configuration
    # --------------------------------------------------
    tactile_img_size = 32

    if obs_type == "oracle":
        # -------- oracle --------
        num_envs = 4096
        if if_use_positional_encoding:
            observation_space = 56
        else:
            observation_space = 15

    elif obs_type == "tactile":
        # -------- tactile --------
        num_envs = 2048

        # Setup tactile sensor config
        # You can set other parameters here for different tactile sensor settings. 
        # Check out the Class TactileSensorCfg for all the parameters.
        tactile_sensor_cfg = TactileSensorCfg(
            tactile_img_size=tactile_img_size,
            if_render_tactile=False,
            sensor_type="right_angle_tactip",  # "tactip" or "right_angle_tactip"
            tactile_depth_enhance_scale=1 / 0.0097,  # scale the depth difference to enhance the deformation visibility. This is computed by getting the maximun depth of difference (our case is 0.0197m, )
            offset_pos=(0, -0.065, 0.0),
            offset_rot=(-0.707, 0.707, 0, 0),
        )
        if "depth" in tactile_sensor_cfg.tactile_image_type:
            if_depth = True
        else:
            if_depth = False

        observation_space = spaces.Box(
            low=float("-inf"),
            high=float("inf"),
            shape=(tactile_img_size, tactile_img_size, 6),
        )

    else:
        raise ValueError(f"Unknown obs_type: {obs_type}")

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
        num_envs=num_envs, env_spacing=1.6,  # replicate_physics=True, clone_in_fabric=True
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

    # Cube
    object_size = 0.08
    object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(object_size, object_size, object_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8, dynamic_friction=.8, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.158, object_size / 2)),  # 0.625, if you add a table, remember to change the z here.
    )

    # Goal sphere
    reset_goal_position_noise = 0.5
    goal_size = 0.01
    traj_n_points = 10
    traj_spacing = 0.025
    traj_max_perturb = 0.02
    traj_type = "straight"  # straight, random_curved
    sub_goal_threshold = 0.025

    goal_cfgs = []
    goal_z_pos = 0.3
    for i in range(traj_n_points):
        goal_cfgs.append(
            RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Goal_{i}",
                spawn=sim_utils.SphereCfg(
                    radius=goal_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=False,
                        disable_gravity=True,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=False
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0),
                        metallic=0.2,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, -0.158 + traj_spacing * i, goal_z_pos),
                ),
            )
        )

    ### Reward function parameters
    obj_goal_pos_dist_scale = 1.0
    obj_goal_orn_dist_scale = 1.0
    tip_obj_orn_dist_scale = 1.0
    obj_tip_pos_dist_scale = 1.0
    terminatie_dist_obj_tip = 0.085  # terminate if the distance between object and tip is larger than this value and no contact
    # robot_cfg
    # robot workframe in scene coordinates, also the initial end-effector pose in the scene/baseframe coordinate! 
    # This should be derived after the robot joint positions are set (the initial_joint_positions below).
    # Then get it by setting if_get_initial_ee_pose = True in the env and print out the ee_pos_b (base frame, scene frame, not world frame)
    if_get_initial_ee_pose = False
    robot_base_in_scene_frame = [-0.55, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # 0.625, if you add a table, remember to change the z here.
    workframe_in_base_frame = [0.55, -0.2, 0.04, 1.0, 0.0, 0.0, 0.0]
    tcp_lims_single = [
        [-0.30, 0.30],                          # x lims
        [0.0, 0.3],                          # y lims
        [-0.0, 0.0],                          # z lims
        [-0.0, 0.0],                          # roll lims
        [-0.0, 0.0],                          # pitch lims
        [-torch.pi / 4, torch.pi / 4]          # yaw lims (-45° to 45°)
    ]  # or torch.double if needed

    # Expand to (num_env, 6, 2)
    # task-specific joint positions
    # The speed is increased to improve exploration efficiency.
    # However, when using tactile observation, a high speed may cause the tactile sensor to miss contacts and cannot recover due to the lack of global obs. 
    robot_ee_speed_scale = [0.05, 0.05, 0.05, 0.25, 0.25, 0.25]  # m/s, rad/s, TODO: finetune the rpy scale

    initial_joint_positions = {
        "base_joint": -0.29460139,
        "shoulder_joint": -2.13012475,
        "elbow_joint": -1.76061934,
        "wrist_1_joint": -0.82164851,
        "wrist_2_joint": 1.5717,
        "wrist_3_joint": 4.41624829,
    }
    robot_cfg = UR5_RA_TACTIP_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UR5_RA_TACTIP_CFG.spawn.replace(
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False  # disable this does not work for entire arms, only work for the root link
            )
        ),
        init_state=UR5_RA_TACTIP_CFG.init_state.replace(
            joint_pos=initial_joint_positions,  # replace the init_state inside the config
            pos=robot_base_in_scene_frame[0:3],
        ),
    )

    ## Contact sensor cfg
    contact_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/tactip_tip_link", history_length=3, update_period=0.005, track_air_time=True
    )

    # set_trace()
    action_scale = 1
    dof_velocity_scale = 0.1

    # reward scales
    goal_dist_reward_scale = 1
    action_penalty_scale = 0.005
    action_rate_penalty_scale = 0.01

