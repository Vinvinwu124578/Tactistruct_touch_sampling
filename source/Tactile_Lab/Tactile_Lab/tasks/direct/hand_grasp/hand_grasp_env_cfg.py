# hand_grasp_env_cfg.py

import torch
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, TiledCameraCfg

from Tactile_Lab.tactile_lab_assets.tactile_lab_assets.robots.tg3_tactile_allegro_hand import (
    TACTILE_ALLEGRO_HAND_ANYROTATE_CFG, 
    TACTILE_ALLEGRO_HAND_ANYROTATE_ROTATED_CFG, 
    TACTILE_ALLEGRO_HAND_ANYROTATE_ROTATED_UPSIDEDOWN_CFG,
    TACTILE_ALLEGRO_HAND_ANYROTATE_ROTATED_UPSIDEDOWN_NO_PINS_CFG,
    TACTILE_ALLEGRO_HAND_ANYROTATE_ROTATED_UPSIDE_NO_PINS_CFG,
    TACTILE_LEAP_HAND_ANYROTATE_ROTATED_UPSIDE_NO_PINS_CFG
)
from Tactile_Lab.utility.tactile_sensor import TactileSensorCfg
from Tactile_Lab.utility.utils import rpy_quat_convert
import numpy as np


@configclass
class HandGraspEnvCfg(DirectRLEnvCfg):

    num_envs = 4096
    if_testing = False
    if_random_hand_pose = True
    if_debug = False
    use_tactile = False
    tactile_img_size = 256
    tactile_update_period = 0.1
    # task_type = "allegro_upside_no_pins_grasp"  # "normal_grasp" or "upsidedown_grasp" or "upsidedown_no_pins_grasp" or "allegro_upside_no_pins_grasp"
    task_type = "leap_upside_no_pins_grasp"
    decimation = 2
    if if_testing:
        episode_length_s = .5
    else:
        episode_length_s = 5.0

    random_hand_rp_range = np.pi  # ±180°
    # Even for grasp generation, Gym requires these
    action_space = 16          # Allegro DOFs
    observation_space = 1      # dummy (we don’t use obs)
    # --------------------------------------------------------------------- #
    # Simulation
    # --------------------------------------------------------------------- #
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=1,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )

    # --------------------------------------------------------------------- #
    # Scene
    # --------------------------------------------------------------------- #
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=0.8,
    )

    # --------------------------------------------------------------------- #
    # Robot
    # --------------------------------------------------------------------- #

    if task_type == "normal_grasp":
        robot_cfg: ArticulationCfg = TACTILE_ALLEGRO_HAND_ANYROTATE_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
        )
        reset_z_threshold = 0.28

    elif task_type == "rotated_grasp":  # upwards facing palm
        robot_cfg: ArticulationCfg = TACTILE_ALLEGRO_HAND_ANYROTATE_ROTATED_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
        )
        reset_z_threshold = 0.28

    elif task_type == "upsidedown_grasp":
        robot_cfg: ArticulationCfg = TACTILE_ALLEGRO_HAND_ANYROTATE_ROTATED_UPSIDEDOWN_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
        )
        reset_z_threshold = 0.26
    elif task_type == "upsidedown_no_pins_grasp":
        robot_cfg: ArticulationCfg = TACTILE_ALLEGRO_HAND_ANYROTATE_ROTATED_UPSIDEDOWN_NO_PINS_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
        )
        reset_z_threshold = 0.26
    elif task_type == "allegro_upside_no_pins_grasp":
        robot_cfg: ArticulationCfg = TACTILE_ALLEGRO_HAND_ANYROTATE_ROTATED_UPSIDE_NO_PINS_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
        )
        reset_z_threshold = 0.02
    elif task_type == "leap_upside_no_pins_grasp":
        robot_cfg: ArticulationCfg = TACTILE_LEAP_HAND_ANYROTATE_ROTATED_UPSIDE_NO_PINS_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
        )
        reset_z_threshold = 0.02

    actuated_joint_names = [
        "joint_0_0", "joint_1_0", "joint_2_0", "joint_3_0",
        "joint_4_0", "joint_5_0", "joint_6_0", "joint_7_0",
        "joint_8_0", "joint_9_0", "joint_10_0", "joint_11_0",
        "joint_12_0", "joint_13_0", "joint_14_0", "joint_15_0",
    ]

    if "leap" in task_type:
        fingertip_body_names = [
            "digitac_tip_thumb_front",
            "digitac_tip_pinky_front",
            "digitac_tip_index_front",
            "digitac_tip_mid_front",
        ]
    else:
        fingertip_body_names = [
            "digitac_tip_link_3",
            "digitac_tip_link_7",
            "digitac_tip_link_11",
            "digitac_tip_link_15",
        ]

    # --------------------------------------------------------------------- #
    # Object
    # --------------------------------------------------------------------- #
    if task_type == "upsidedown_no_pins_grasp":
        object_pose_in_handframe =[0.0, 0.0, -0.10, 1.0, 0.0, 0.0, 0.0]  # (x,y,z, qw,qx,qy,qz)
    elif task_type == "allegro_upside_no_pins_grasp" or task_type == "leap_upside_no_pins_grasp":
        object_pose_in_handframe =[0.0, 0.0, 0.125, 1.0, 0.0, 0.0, 0.0]  # (x,y,z, qw,qx,qy,qz)
    
    object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CapsuleCfg(
            radius=0.035, height=0.007,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=.15),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=0.5, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.15),  #  Not used, will be overwritten by grasp generation script to spawn in hand frame
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    if "leap" in task_type:
        fingertip_contact_sensors_cfgs: dict = {
            "digitac_tip_thumb_front_contact_sensor": ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/digitac_tip_thumb_front",
                history_length=3,
                update_period=0.005,
                track_air_time=True,
            ),
            "digitac_tip_pinky_front_contact_sensor": ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/digitac_tip_pinky_front",
                history_length=3,
                update_period=0.005,
                track_air_time=True,
            ),
            "digitac_tip_index_front_contact_sensor": ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/digitac_tip_index_front",
                history_length=3,
                update_period=0.005,
                track_air_time=True,
            ),
            "digitac_tip_mid_front_contact_sensor": ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/digitac_tip_mid_front",
                history_length=3,
                update_period=0.005,
                track_air_time=True,
            ),
        }
    else:
        fingertip_contact_sensors_cfgs: dict = {
            "digitac_tip_link_3_contact_sensor": ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/digitac_tip_link_3",
                history_length=3,
                update_period=0.005,
                track_air_time=True,
            ),
            "digitac_tip_link_7_contact_sensor": ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/digitac_tip_link_7",
                history_length=3,
                update_period=0.005,
                track_air_time=True,
            ),
            "digitac_tip_link_11_contact_sensor": ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/digitac_tip_link_11",
                history_length=3,
                update_period=0.005,
                track_air_time=True,
            ),
            "digitac_tip_link_15_contact_sensor": ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/digitac_tip_link_15",
                history_length=3,
                update_period=0.005,
                track_air_time=True,
            ),
        }

    ## tactile
    if use_tactile:
        if "leap" in task_type:
            camera_link_indices = [0, 1, 2, 3]  # hardcoded for Leap hand, corresponds to fingertip links
            sensor_type = "leap_hand_tactip"
            offset_rpy = [3.14, 3.14, 0]
            offset_rot = rpy_quat_convert(rpy=offset_rpy)
        else:
            camera_link_indices = [3, 7, 11, 15]  # hardcoded for Allegro hand, corresponds to fingertip links
            sensor_type = "hand_finger"
            offset_rpy = [-1.57, 0, 0]
            offset_rot = rpy_quat_convert(rpy=offset_rpy)
        tactile_sensors_num = len(camera_link_indices)
        tactile_sensor_cfgs = []
        for cam_idx, link_idx in enumerate(camera_link_indices):
            if "leap" in task_type:
                prim_path = f"/World/envs/env_.*/Robot/link_{link_idx}_camera/tactile_cam"
            else:
                prim_path = f"/World/envs/env_.*/Robot/link_{link_idx}_tcp/tactile_cam"
            tactile_sensor_cfgs.append(
                TactileSensorCfg(
                    if_render_tactile=False,
                    sensor_type=sensor_type,
                    if_save_tactile_reference_image=False,
                    tactile_img_size=tactile_img_size,
                    tactile_image_type=["depth", "tactile_flow"],
                    tactile_depth_enhance_scale=1 / 0.0097,
                    focal_length=7,
                    offset_pos=(0.0, 0.0, 0.0),
                    offset_rot=offset_rot,
                    update_period=tactile_update_period,
                    prim_path=prim_path,
                    idx=cam_idx  # for multi-sensor setups (e.g. bipush), can be used to differentiate sensors in rendering
                )
            )
        if_depth = "depth" in tactile_sensor_cfgs[0].tactile_image_type and "depth" in tactile_sensor_cfgs[1].tactile_image_type
                
    # --------------------------------------------------------------------- #
    # Grasp generation parameters
    # --------------------------------------------------------------------- #
    min_tip_force = 0.01
    grasp_settle_steps = 50
    fingertip_dist_thresh = 0.10
    min_contact_fingers = 3

    if if_random_hand_pose:
        grasp_cache_path = "cache/isaaclab_allegro_" + str(task_type) + ".npy"
    else:
        grasp_cache_path = "cache/isaaclab_allegro_" + str(task_type) + "_fixed_hand_pose.npy"
    act_moving_average = 1.0
    total_collect_grasp_num = 100000
