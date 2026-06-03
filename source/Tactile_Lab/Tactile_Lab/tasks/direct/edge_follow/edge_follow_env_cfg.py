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
# from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG, UR5_TACTIP_CFG   # isort:skip
from Tactile_Lab.tactile_lab_assets.tactile_lab_assets.robots.tg3_ur5 import UR5_TACTIP_CFG   # isort:skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns, TiledCameraCfg
import torch
from gymnasium import spaces
from importlib import resources
from pathlib import Path
import os
from Tactile_Lab.utility.tactile_sensor import TactileSensorCfg
from Tactile_Lab.utility.utils import rpy_quat_convert


from ipdb import set_trace
def get_tactile_lab_assets_root() -> Path:
    for p in Path(__file__).resolve().parents:
        candidate = p.parent / "Tactile_Lab_External_Assets"
        if candidate.exists():
            return candidate
    raise RuntimeError("Could not locate Tactile_Lab_External_Assets")
ASSET_ROOT = get_tactile_lab_assets_root()
print(' [Tactile_Lab] Tactile Lab Assets root path: ', ASSET_ROOT)

@configclass
class EdgeFollowEnvCfg(DirectRLEnvCfg):

    obs_type = "tactile"   # "oracle" | "tactile"
    episode_length_s = 6
    decimation = 2
    action_space = 2
    if_use_positional_encoding = True
    tactile_sensor_cfg = None
    tactile_img_size = 32

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
        num_envs = 2048
        edge_stop_distance = 0.01
        success_threshold = 0.015
        cam_rpy = [3.14, 3.14, 1.57]  # for using franka_panda_tactip_flip
        cam_quat = rpy_quat_convert(rpy=cam_rpy)  # [w, x, y, z]
        tactile_sensor_cfg = TactileSensorCfg(
            if_render_tactile=False,
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
    if tactile_sensor_cfg is not None and tactile_sensor_cfg.if_save_tactile_reference_image:
        edge_offset_for_save_image = 1.0
    else:
        edge_offset_for_save_image = 0.0
    edge_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Edge",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ASSET_ROOT, "Robots/tg3_asset/long_edge.usd"),
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

    # robot_cfg
    workframe_scene = [0.65, 0, 0.022, 3.14, 0.0, 0.0]
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

