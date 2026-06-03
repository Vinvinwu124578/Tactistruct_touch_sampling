# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from Tactile_Lab.tactile_lab_assets.tactile_lab_assets.robots.tg3_ur5 import UR5_TACTIP_CFG, BITOUCH_UR5_RA_TACTIP_CFG   # isort:skip
from ipdb import set_trace

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg, ContactSensorCfg, TiledCameraCfg
import torch
from gymnasium import spaces
from Tactile_Lab.utility.tactile_sensor import TactileSensorCfg
from Tactile_Lab.utility.utils import rpy_quat_convert


@configclass
class BiPushEnvCfg(DirectRLEnvCfg):
    obs_type = "oracle"   # "oracle" | "tactile"
    if_use_positional_encoding = False
    if_render_tactile = False
    # --------------------------------------------------
    # General env settings
    # --------------------------------------------------
    episode_length_s = 15
    decimation = 2
    action_space = 6          # 6D pose control (xyz + rpy)
    clip_actions = 1.0
    state_space = 0
    robot_num = 2
    cam_num = robot_num
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
            observation_space = 21

    elif obs_type == "tactile":
        # -------- tactile --------
        num_envs = 2048

        # Setup tactile sensor config
        # You can set other parameters here for different tactile sensor settings. 
        # Check out the Class TactileSensorCfg for all the parameters.
        # Setup tactile sensor configs for both robots
        offset_rpy = [-1.57, 0, -1.57]  # for using franka_panda_tactip_flip
        offset_rot = rpy_quat_convert(rpy=offset_rpy)  # [w, x, y, z]

        tactile_sensor_cfgs = []

        for i in range(cam_num):
            tactile_sensor_cfgs.append(
                TactileSensorCfg(
                    tactile_img_size=tactile_img_size,
                    if_render_tactile=if_render_tactile,
                    if_save_tactile_reference_image=False,
                    prim_path=f"/World/envs/env_.*/Robot_{i}/tcp_link/tactile_cam",
                    sensor_type="bitouch_forward_right_angle_tactip",
                    tactile_depth_enhance_scale=1 / 0.0097,
                    offset_pos=(-0.065, 0.0, 0.0),
                    offset_rot=offset_rot,
                    idx=i,
                )
            )

        if_depth = "depth" in tactile_sensor_cfgs[0].tactile_image_type and "depth" in tactile_sensor_cfgs[1].tactile_image_type

        # observation_space = spaces.Box(
        #     low=float("-inf"),
        #     high=float("inf"),
        #     shape=(15, tactile_img_size, 2*tactile_img_size, 1),  # TODO: seems does not affect the obs space in env, need to check the env code to make sure the obs is concatenated correctly.
        # )
        observation_space = {
            "image": [1, tactile_img_size, 2 * tactile_img_size],
            "feature": 15,
        }
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
    object_size = 0.1
    object_length = 0.8
    object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(object_size, object_length, object_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8, dynamic_friction=.8, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.20 + object_size / 2, 0.0, object_size / 2)),  # pos in world frame, need to determine the ee init pos first, if you add a table, remember to change the z here.
    )

    # Goal sphere
    reset_goal_position_noise = 0.5
    goal_size = 0.01
    traj_n_points = 10
    traj_spacing = 0.025
    traj_max_perturb = 0.05
    traj_type = "straight"  # straight, random_curved
    sub_goal_threshold = 0.035

    # goal_cfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Goal",
    #     spawn=sim_utils.SphereCfg(
    #         radius=goal_size,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.158, 0.5)),  # 0.625, if you add a table, remember to change the z here.
    # )
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
    obj_tip_pos_dist_scale = 1000.0
    terminatie_dist_obj_tip = 0.085  # terminate if the distance between object and tip is larger than this value and no contact
    # robot_cfg
    # robot workframe in scene coordinates, also the initial end-effector pose in the scene/baseframe coordinate! 
    # This should be derived after the robot joint positions are set (the initial_joint_positions below).
    # Then get it by setting if_get_initial_ee_pose = True in the env and print out the ee_pos_b (base frame, scene frame, not world frame)
    if_get_initial_ee_pose = False
    robot_base_offset = 0.25 
    robot_base_in_scene_frame_0 = [-0.55, -robot_base_offset, 0.0, 1.0, 0.0, 0.0, 0.0]  # 0.625, if you add a table, remember to change the z here.
    robot_base_in_scene_frame_1 = [-0.55, robot_base_offset, 0.0, 1.0, 0.0, 0.0, 0.0]  # Added for second robot
    ee_init_pose_base = [0.35, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]  # obtained from if_get_initial_ee_pose
    workframe_in_scene_frame = [-0.2, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]
    tcp_lims_robot_0 = [
        [-0.0, 0.3],
        [-robot_base_offset - 0.1, -robot_base_offset + 0.1],
        [-0.0, 0.0],
        [-0.0, 0.0],
        [-0.0, 0.0],
        [-torch.pi / 4, torch.pi / 4],
    ]

    tcp_lims_robot_1 = [
        [-0.0, 0.3],
        [ robot_base_offset - 0.1, robot_base_offset + 0.1],
        [-0.0, 0.0],
        [-0.0, 0.0],
        [-0.0, 0.0],
        [-torch.pi / 4, torch.pi / 4],
    ]

    # Expand to (num_env, 6, 2)
    # task-specific joint positions
    # The speed is increased to improve exploration efficiency.
    # However, when using tactile observation, a high speed may cause the tactile sensor to miss contacts and cannot recover due to the lack of global obs. 
    robot_ee_speed_scale = [0.2, 0.2, 0.2, 1, 1, 1]  # m/s, rad/s, TODO: finetune the rpy scale

    initial_joint_positions = {
        "base_joint": 0.413888,
        "shoulder_joint": -1.466998,
        "elbow_joint": -2.720950,
        "wrist_1_joint": -0.524851,
        "wrist_2_joint": 1.571846,
        "wrist_3_joint": -1.157706,
    }

    robot_cfg_0 = BITOUCH_UR5_RA_TACTIP_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_0",
        spawn=BITOUCH_UR5_RA_TACTIP_CFG.spawn.replace(
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False  # only affects root link
            )
        ),
        init_state=BITOUCH_UR5_RA_TACTIP_CFG.init_state.replace(
            joint_pos=initial_joint_positions,
            pos=robot_base_in_scene_frame_0[0:3],
        ),
    )

    robot_cfg_1 = BITOUCH_UR5_RA_TACTIP_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_1",
        spawn=BITOUCH_UR5_RA_TACTIP_CFG.spawn.replace(
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False  # only affects root link
            )
        ),
        init_state=BITOUCH_UR5_RA_TACTIP_CFG.init_state.replace(
            joint_pos=initial_joint_positions,
            pos=robot_base_in_scene_frame_1[0:3],
        ),
    )
    # Contact sensor for robot 0
    contact_sensor_cfg_0 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_0/tactip_tip_link",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # Contact sensor for robot 1
    contact_sensor_cfg_1 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_1/tactip_tip_link",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # set_trace()
    action_scale = 1
    dof_velocity_scale = 0.1

    # reward scales

    goal_dist_reward_scale = 1
    action_penalty_scale = 0.005
    action_rate_penalty_scale = 0.01

