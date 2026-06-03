# allegro_gait_gen_env_cfg.py

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, TiledCameraCfg
from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg

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


@configclass
class HandGaitEnvCfg(DirectRLEnvCfg):

    obs_type = "oracle"  # "oracle" or "tactile" or "dense_tactile"
    tactile_img_size = 256
    if_testing = False
    priv_obs_norm = True
    if_frames_vis = False
    if_tactile_vis = False  # for when using oracle obs but want to visualize tactile observations for debugging
    if_force_disturbance = True
    if_random_hand_pose = True
    if_random_joint_impulse = False
    update_period = 0.01 if if_tactile_vis else 0.1  # how often to update tactile obs and privileged obs (if using tactile obs, we also update priv obs at the same frequency since they are used together in RSL-RL)
    obs_hist_len = 15
    if_use_positional_encoding = True  # for priv obj position
    ctrl_mode = "position"  # "position" or "torque"
    p_gain = 3
    d_gain = 0.1

    # task_type = "allegro_upside_no_pins_grasp"  # "normal_grasp" or "upsidedown_grasp" or "upsidedown_no_pins_grasp" or "allegro_upside_no_pins_grasp"
    task_type = "leap_upside_no_pins_grasp"
    decimation = 2

    # dense_tactile
    contact_dim_per_finger = 4  # contact position x y + contact flag + net force
    min_tip_force = 0.01
    if_use_low_dim_contact_obs = False

    if obs_type == "oracle":
        num_envs = 8192  # 4096, 8192, 16384, 32768
        observation_space = 21   # not used with RSL-RL
    if obs_type == "tactile" or if_tactile_vis:
        if not if_tactile_vis:
            num_envs = 1024
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
                    update_period=update_period,
                    prim_path=prim_path,
                    idx=cam_idx  # for multi-sensor setups (e.g. bipush), can be used to differentiate sensors in rendering
                )
            )
        if_depth = "depth" in tactile_sensor_cfgs[0].tactile_image_type and "depth" in tactile_sensor_cfgs[1].tactile_image_type
        observation_space = {  # not used with RSL-RL
            "image": [1, tactile_img_size, 2 * tactile_img_size],
            "feature": 15,
        }
    # else:
    #     raise ValueError(f"Unknown obs_type: {obs_type}")

    # Force disturbance parameters
    if if_force_disturbance:
        disturbance_axis = ["x", "y", "z"]  # "z"
        force_range = (-.75, .75)
        torque_range = (-0.008, 0.008)

    if if_testing:
        episode_length_s = 4.0
    else:
        episode_length_s = 10.0  # 600 time steps at 2xdt=0.016s
    # Even for gait generation, Gym requires these
    action_space = 16          # Allegro DOFs
    observation_space = 99      # dummy (we don’t use obs)
    state_space = 0

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
        env_spacing=0.6,
        replicate_physics=False,
    )

    # Randomization
    @configclass
    class EventsCfg:

        object_scale_usd = EventTerm(
            func=mdp.randomize_rigid_body_scale,
            mode="prestartup",   # MUST be prestartup
            params={
                "asset_cfg": SceneEntityCfg("object"),
                #  "scale_range": (0.975, 1.025),   # isotropic scale
                 "scale_range": (0.9, 1.3),   # isotropic scale
                # optional:
                # "relative_child_path": "mesh"
            },
        )

        object_physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            min_step_count_between_reset=720,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "static_friction_range": (0.3, 3.0),
                "dynamic_friction_range": (0.3, 3.0),
                "restitution_range": (0.0, 0.2),
                "num_buckets": 250,
            },
        )

        object_scale_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            min_step_count_between_reset=720,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "mass_distribution_params": (0.01, 0.25),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        object_com = EventTerm(
            func=mdp.randomize_rigid_body_com_non_accumulation,
            min_step_count_between_reset=720,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names=".*"),
                "com_range": {
                    "x": (-0.015, 0.015),
                    "y": (-0.015, 0.015),
                    "z": (-0.015, 0.015),
                },
            },
        )

    # Register events
    events: EventsCfg = EventsCfg()

    # --------------------------------------------------------------------- #
    # Robot
    # --------------------------------------------------------------------- #
    if task_type == "normal_grasp":
        robot_cfg: ArticulationCfg = TACTILE_ALLEGRO_HAND_ANYROTATE_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
        )
        reset_z_threshold = 0.28
        grasp_cache_path = (
            "/home/bourne/Tactile_Lab/scripts/cache/"
            "isaaclab_allegro_grasps.npy"
        )

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
        reset_anchor_dist_threshold = 0.15  # if the object is more than this distance away from the hand anchor point, we consider it as dropped and reset the environment
        grasp_cache_path = (
            "/home/bourne/Tactile_Lab/scripts/cache/"
            "isaaclab_allegro_upsidedown_grasp.npy"
        )
    elif task_type == "upsidedown_no_pins_grasp":
        robot_cfg: ArticulationCfg = TACTILE_ALLEGRO_HAND_ANYROTATE_ROTATED_UPSIDEDOWN_NO_PINS_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
        )
        object_pose_in_handframe = [0.0, 0.0, -0.10, 1.0, 0.0, 0.0, 0.0]  # (x,y,z, qw,qx,qy,qz)
        reset_z_threshold = 0.26
        reset_anchor_dist_threshold = 0.15  # if the object is more than this distance away from the hand anchor point, we consider it as dropped and reset the environment
        if if_random_hand_pose:
            grasp_cache_path = (
                "/home/bourne/Tactile_Lab/scripts/cache/"
                "isaaclab_allegro_upsidedown_no_pins_grasp.npy"
            )
        else:
            grasp_cache_path = (
                "/home/bourne/Tactile_Lab/scripts/cache/"
                "isaaclab_allegro_upsidedown_no_pins_grasp_fixed_hand_pose.npy"
            )
    elif task_type == "allegro_upside_no_pins_grasp":
        robot_cfg: ArticulationCfg = TACTILE_ALLEGRO_HAND_ANYROTATE_ROTATED_UPSIDE_NO_PINS_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
        )
        object_pose_in_handframe = [0.0, 0.0, 0.125, 1.0, 0.0, 0.0, 0.0]  # (x,y,z, qw,qx,qy,qz)
        reset_z_threshold = 0.34
        reset_anchor_dist_threshold = 0.08  # if the object is more than this distance away from the hand anchor point, we consider it as dropped and reset the environment

        if if_random_hand_pose:
            grasp_cache_path = (
                "/home/bourne/Tactile_Lab/scripts/cache/"
                "isaaclab_allegro_upside_no_pins_grasp.npy"
            )
        else:
            grasp_cache_path = (
                "/home/bourne/Tactile_Lab/scripts/cache/"
                "isaaclab_allegro_upside_no_pins_grasp_fixed_hand_pose.npy"
            )
    elif task_type == "leap_upside_no_pins_grasp":
        robot_cfg: ArticulationCfg = TACTILE_LEAP_HAND_ANYROTATE_ROTATED_UPSIDE_NO_PINS_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
        )
        object_pose_in_handframe = [0.0, 0.0, 0.125, 1.0, 0.0, 0.0, 0.0]  # (x,y,z, qw,qx,qy,qz)
        reset_z_threshold = 0.34
        reset_anchor_dist_threshold = 0.08  # if the object is more than this distance away from the hand anchor point, we consider it as dropped and reset the environment

        if if_random_hand_pose:
            grasp_cache_path = (
                "/home/bourne/Tactile_Lab/scripts/cache/"
                "isaaclab_allegro_upside_no_pins_grasp.npy"
            )
        else:
            grasp_cache_path = (
                "/home/bourne/Tactile_Lab/scripts/cache/"
                "isaaclab_allegro_upside_no_pins_grasp_fixed_hand_pose.npy"
            )

            

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
    object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CapsuleCfg(
            radius=0.035, height=0.007,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),  # It will be scaled by the event randomizer, so the initial value doesn't matter much
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.5, restitution=0.0, compliant_contact_stiffness=0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.395),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    #--------------------------------------------------------------------- #
    # Contact Sensors
    #---------------------------------------------------------------------- #
    if if_use_low_dim_contact_obs:
        fingertip_contact_sensors_cfgs: dict = {
            "digitac_tip_link_3_contact_sensor": ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/digitac_tip_link_3",
                track_pose=True,
                track_contact_points=True,
                # debug_vis=True,
                history_length=3,
                update_period=update_period,
                track_air_time=True,
                filter_prim_paths_expr=["/World/envs/env_.*/Object"],
                max_contact_data_count_per_prim=16,  # >=1 required
            ),
            "digitac_tip_link_7_contact_sensor": ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/digitac_tip_link_7",
                track_pose=True,
                track_contact_points=True,
                history_length=3,
                update_period=update_period,
                track_air_time=True,
                filter_prim_paths_expr=["/World/envs/env_.*/Object"],
                max_contact_data_count_per_prim=16,  # >=1 required
            ),
            "digitac_tip_link_11_contact_sensor": ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/digitac_tip_link_11",
                track_pose=True,
                track_contact_points=True,
                history_length=3,
                update_period=update_period,
                track_air_time=True,
                filter_prim_paths_expr=["/World/envs/env_.*/Object"],
                max_contact_data_count_per_prim=16,  # >=1 required
            ),
            "digitac_tip_link_15_contact_sensor": ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/digitac_tip_link_15",
                track_pose=True,
                track_contact_points=True,
                history_length=3,
                update_period=update_period,
                track_air_time=True,
                filter_prim_paths_expr=["/World/envs/env_.*/Object"],
                max_contact_data_count_per_prim=16,  # >=1 required
            ),
        }

    # --------------------------------------------------------------------- #
    # Gait generation parameters
    # --------------------------------------------------------------------- #

    act_moving_average = 0.75
    total_collect_gait_num = 10000

    joint_noise_scale = 0.02
    obs_hist_total_len = 15
    actions_scale = 1.0
    action_clip = 1.0
    max_position_delta = 0.04
    angvel_clip_range = 0.5
    torque_limit = 0.5
    # Reward scales
    obj_rotate_reward_scale = 1.0
    obj_linear_vel_penalty_scale = -0.3
    pose_diff_penalty_scale = -0.3
    if ctrl_mode == "position":
        pos_error_penalty_scale = -0.3
        work_penalty_scale = -0.001
    elif ctrl_mode == "torque":
        pos_error_penalty_scale = -0.1
        work_penalty_scale = -2.0
    target_delta_penalty_scale = 0.0
