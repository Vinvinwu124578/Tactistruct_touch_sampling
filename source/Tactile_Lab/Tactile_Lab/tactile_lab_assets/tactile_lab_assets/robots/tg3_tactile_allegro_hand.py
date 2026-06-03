import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import math
import os
from pathlib import Path


def get_tactile_lab_assets_root() -> Path:
    for p in Path(__file__).resolve().parents:
        candidate = p.parent / "Tactile_Lab_External_Assets"
        if candidate.exists():
            return candidate
    raise RuntimeError("Could not locate Tactile_Lab_External_Assets")


ASSET_ROOT = get_tactile_lab_assets_root()
print(' [Tactile_Lab] Tactile Lab Assets root path: ', ASSET_ROOT)


hand_config = {
    # for anyrotate task
    'no_palm_support_rotated': {
        'init_pos': (0.0, 0.0, 0.25),
        'init_rot': (1, 0, 0, 0),
        'joint_pos':
            {
            "joint_0_0":   0.0578,
            "joint_1_0":   1.3368,
            "joint_2_0":   0.4102,
            "joint_3_0":   0.0521,
            "joint_12_0":  1.2420,
            "joint_13_0":  0.9322,
            "joint_14_0":  0.3797,
            "joint_15_0":  0.3927,
            "joint_4_0":  -0.0000,
            "joint_5_0":   1.0312,
            "joint_6_0":   0.3136,
            "joint_7_0":   0.1278,
            "joint_8_0":  -0.0337,
            "joint_9_0":   1.4387,
            "joint_10_0":  0.2364,
            "joint_11_0":  0.2697,
        },
    'usd_path': os.path.join(ASSET_ROOT, "Robots/tg3_asset/allegro_digitac_v2_inw_90_45_90_no_palm_rotated.usd"),
    },
    'no_palm_support_rotated_upsidedown': {
        'init_pos': (0.0, 0.0, 0.45),
        'init_rot': (1, 0, 0, 0),
        'joint_pos':
            {
            "joint_0_0":   0.0578,
            "joint_1_0":   1.3368,
            "joint_2_0":   0.4102,
            "joint_3_0":   0.0521,
            "joint_12_0":  1.2420,
            "joint_13_0":  0.9322,
            "joint_14_0":  0.3797,
            "joint_15_0":  0.3927,
            "joint_4_0":  -0.0000,
            "joint_5_0":   1.0312,
            "joint_6_0":   0.3136,
            "joint_7_0":   0.1278,
            "joint_8_0":  -0.0337,
            "joint_9_0":   1.4387,
            "joint_10_0":  0.2364,
            "joint_11_0":  0.2697,
        },
    'usd_path': os.path.join(ASSET_ROOT, "Robots/tg3_asset/allegro_digitac_v2_inw_90_45_90_no_palm_rotated_upsidedown.usd"),
    },
    'no_palm_support_rotated_upsidedown_no_pins': {
        'init_pos': (0.0, 0.0, 0.45),
        'init_rot': (1, 0, 0, 0),
        'joint_pos':
            {
            "joint_0_0":   0.0578,
            "joint_1_0":   1.3368,
            "joint_2_0":   0.4102,
            "joint_3_0":   0.0521,
            "joint_12_0":  1.2420,
            "joint_13_0":  0.9322,
            "joint_14_0":  0.3797,
            "joint_15_0":  0.3927,
            "joint_4_0":  -0.0000,
            "joint_5_0":   1.0312,
            "joint_6_0":   0.3136,
            "joint_7_0":   0.1278,
            "joint_8_0":  -0.0337,
            "joint_9_0":   1.4387,
            "joint_10_0":  0.2364,
            "joint_11_0":  0.2697,
        },
    'usd_path': os.path.join(ASSET_ROOT, "Robots/tg3_asset/allegro_digitac_v2_inw_90_45_90_no_palm_rotated_upsidedown_no_pins.usd"),
    },
    'no_palm_support_rotated_upside_no_pins': {
        'init_pos': (0.0, 0.0, 0.25),
        'init_rot': (1, 0, 0, 0),
        'joint_pos':
            {
            "joint_0_0":   0.0578,
            "joint_1_0":   1.3368,
            "joint_2_0":   0.4102,
            "joint_3_0":   0.0521,
            "joint_12_0":  1.2420,
            "joint_13_0":  0.9322,
            "joint_14_0":  0.3797,
            "joint_15_0":  0.3927,
            "joint_4_0":  -0.0000,
            "joint_5_0":   1.0312,
            "joint_6_0":   0.3136,
            "joint_7_0":   0.1278,
            "joint_8_0":  -0.0337,
            "joint_9_0":   1.4387,
            "joint_10_0":  0.2364,
            "joint_11_0":  0.2697,
        },
    'usd_path': os.path.join(ASSET_ROOT, "Robots/tg3_asset/allegro_digitac_v2_inw_90_45_90_no_palm_rotated_upside_no_pins.usd"),
    },
    'leap_hand_upside_no_pins': {
        'init_pos': (0.0, 0.0, 0.25),
        'init_rot': (1, 0, 0, 0),
        'joint_pos':
            {
            "joint_0_0":   0.0578,
            "joint_1_0":   1.3368,
            "joint_2_0":   0.4102,
            "joint_3_0":   0.0521,
            "joint_12_0":  1.2420,
            "joint_13_0":  0.9322,
            "joint_14_0":  0.3797,
            "joint_15_0":  0.3927,
            "joint_4_0":  -0.0000,
            "joint_5_0":   1.0312,
            "joint_6_0":   0.3136,
            "joint_7_0":   0.1278,
            "joint_8_0":  -0.0337,
            "joint_9_0":   1.4387,
            "joint_10_0":  0.2364,
            "joint_11_0":  0.2697,
        },
    'usd_path': os.path.join(ASSET_ROOT, "Robots/tg3_asset/robot_tip2_shell.usd"),
    },
    'no_palm_support': {
        'init_pos': (0.0, 0.0, 0.25),
        'init_rot': (-0.707, 0, 0.707, 0),
        'joint_pos':
            {
            "joint_0_0":   0.0578,
            "joint_1_0":   1.3368,
            "joint_2_0":   0.4102,
            "joint_3_0":   0.0521,
            "joint_12_0":  1.2420,
            "joint_13_0":  0.9322,
            "joint_14_0":  0.3797,
            "joint_15_0":  0.3927,
            "joint_4_0":  -0.0000,
            "joint_5_0":   1.0312,
            "joint_6_0":   0.3136,
            "joint_7_0":   0.1278,
            "joint_8_0":  -0.0337,
            "joint_9_0":   1.4387,
            "joint_10_0":  0.2364,
            "joint_11_0":  0.2697,
        },
    'usd_path': os.path.join(ASSET_ROOT, "Robots/tg3_asset/allegro_digitac_v2_inw_90_45_90_no_palm.usd"),
    },

    # For cube_repose task
    'palm_support':{
        'init_pos': (0.0, 0.0, 0.5),
        'init_rot': (-6.7559e-01, 6.1758e-08, 7.3728e-01, 6.1758e-08),
        'joint_pos':
            {
            "^(?!joint_12_0).*": 0.0,  # all joints except thumb base
            "joint_12_0": 0.28,        # thumb abduction
            },
        'usd_path': os.path.join(ASSET_ROOT, "Robots/tg3_asset/allegro_digitac_v2_inw_90_45_90_with_palm.usd"),
        },
}


def make_allegro_hand_cfg(
    palm_config: str = "palm_support",
    activate_contact_sensors: bool = False,
) -> ArticulationCfg:
    """
    IsaacLab articulation config for Allegro Hand (USD-safe names).

    Assumes joints are named:
        joint_0_0 ... joint_15_0
    Ordered by finger:
        Index  : 0–3
        Middle : 4–7
        Ring   : 8–11
        Thumb  : 12–15
    """

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=hand_config[palm_config]['usd_path'],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                enable_gyroscopic_forces=False,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=64 / math.pi * 180.0,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            activate_contact_sensors=activate_contact_sensors,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            # collision_props=sim_utils.CollisionPropertiesCfg(
            #     contact_offset=0.001,
            #     rest_offset=-0.0005,
            # ),
        ),

        # -----------------------
        # Initial joint posture
        # -----------------------     robot_base_in_scene_frame_pose = [0.0, -0.0, 0.5, -0.7071, 0.0, 0.7071, 0.0]  # x, y, z, w, x, y, z

        init_state=ArticulationCfg.InitialStateCfg(
            pos=hand_config[palm_config]['init_pos'],
            rot=hand_config[palm_config]['init_rot'],
            joint_pos=hand_config[palm_config]['joint_pos'],
        ),

        # -----------------------
        # Actuators
        # -----------------------
        actuators={
            "index": ImplicitActuatorCfg(
                joint_names_expr=[
                    "joint_0_0",
                    "joint_1_0",
                    "joint_2_0",
                    "joint_3_0",
                ],
                effort_limit_sim=0.5,
                stiffness=3.0,
                damping=0.1,
            ),
            "middle": ImplicitActuatorCfg(
                joint_names_expr=[
                    "joint_4_0",
                    "joint_5_0",
                    "joint_6_0",
                    "joint_7_0",
                ],
                effort_limit_sim=0.5,
                stiffness=3.0,
                damping=0.1,
            ),
            "ring": ImplicitActuatorCfg(
                joint_names_expr=[
                    "joint_8_0",
                    "joint_9_0",
                    "joint_10_0",
                    "joint_11_0",
                ],
                effort_limit_sim=0.5,
                stiffness=3.0,
                damping=0.1,
            ),
            "thumb": ImplicitActuatorCfg(
                joint_names_expr=[
                    "joint_12_0",
                    "joint_13_0",
                    "joint_14_0",
                    "joint_15_0",
                ],
                effort_limit_sim=0.5,
                stiffness=3.0,
                damping=0.1,
            ),
        },
    )


# -------------------------------------------------
# Concrete configs (same style as your UR5 example)
# -------------------------------------------------

TACTILE_ALLEGRO_HAND_CFG = make_allegro_hand_cfg(
    activate_contact_sensors=False,
)

TACTILE_ALLEGRO_HAND_ANYROTATE_CFG = make_allegro_hand_cfg(
    palm_config="no_palm_support",
    activate_contact_sensors=True,
)

TACTILE_ALLEGRO_HAND_ANYROTATE_ROTATED_CFG = make_allegro_hand_cfg(
    palm_config="no_palm_support_rotated",
    activate_contact_sensors=True,
)

TACTILE_ALLEGRO_HAND_ANYROTATE_ROTATED_UPSIDEDOWN_CFG = make_allegro_hand_cfg(
    palm_config="no_palm_support_rotated_upsidedown",
    activate_contact_sensors=True,
)

TACTILE_ALLEGRO_HAND_ANYROTATE_ROTATED_UPSIDEDOWN_NO_PINS_CFG = make_allegro_hand_cfg(
    palm_config="no_palm_support_rotated_upsidedown_no_pins",
    activate_contact_sensors=True,
)

TACTILE_ALLEGRO_HAND_ANYROTATE_ROTATED_UPSIDE_NO_PINS_CFG = make_allegro_hand_cfg(
    palm_config="no_palm_support_rotated_upside_no_pins",
    activate_contact_sensors=True,
)

TACTILE_LEAP_HAND_ANYROTATE_ROTATED_UPSIDE_NO_PINS_CFG = make_allegro_hand_cfg(
    palm_config="leap_hand_upside_no_pins",
    activate_contact_sensors=True,
)
