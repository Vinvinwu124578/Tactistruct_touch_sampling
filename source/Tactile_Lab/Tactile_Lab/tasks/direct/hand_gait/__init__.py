# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym


from . import agents

##
# Register Gym environments.
##

###############################
# Teacher#
###############################

#Extrinsics Stage 1 Oracle Teacher#
gym.register(
    id="Hand-Gait-Extrinsics-S1-Oracle-v0",
    entry_point=f"{__name__}.hand_gait_env:HandGaitEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hand_gait_env_cfg:HandGaitEnvCfg",  # TODO: if have a separate cfg class python file, remember to change the path
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HandGaitExtrinsicsPPORunnerCfg",  # rsl_rl, oracle teacher
    },
)

#Extrinsics Stage 2 Proprio#
gym.register(
    id="Hand-Gait-Extrinsics-S2-Proprio-Distillation-v0",
    entry_point=f"{__name__}.hand_gait_distillation_env:HandGaitDistillationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hand_gait_distillation_env_cfg:HandGaitDistillationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HandGaitExtrinsicsProprioDistillationRunnerCfg",
    },
)

#Extrinsics Stage 2 Proprio and Low Dim Contact#
gym.register(
    id="Hand-Gait-Extrinsics-S2-Proprio-Low-Dim-Contact-Distillation-v0",
    entry_point=f"{__name__}.hand_gait_distillation_env:HandGaitDistillationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hand_gait_distillation_env_cfg:HandGaitDistillationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HandGaitExtrinsicsProprioLowDimContactDistillationRunnerCfg",
    },
)

#Extrinsics Stage 2 Multimodal#
gym.register(
    id="Hand-Gait-Extrinsics-S2-MultiModal-Distillation-v0",
    entry_point=f"{__name__}.hand_gait_distillation_env:HandGaitDistillationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hand_gait_distillation_env_cfg:HandGaitDistillationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HandGaitExtrinsicsMultiModalDistillationRunnerCfg",
    },
)

#Extrinsics Stage 2 Proprio, Multimodal and Low Dim Contact#
gym.register(
    id="Hand-Gait-Extrinsics-S2-MultiModal-Low-Dim-Contact-Distillation-v0",
    entry_point=f"{__name__}.hand_gait_distillation_env:HandGaitDistillationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hand_gait_distillation_env_cfg:HandGaitDistillationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HandGaitExtrinsicsMultiModalLowDimContactDistillationRunnerCfg",
    },
)

#Oracle Teacher#
gym.register(
    id="Hand-Gait-Oracle-v0",
    entry_point=f"{__name__}.hand_gait_env:HandGaitEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hand_gait_env_cfg:HandGaitEnvCfg",  # TODO: if have a separate cfg class python file, remember to change the path
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",  # rl_games, for both oracle and tactile, but need to modify the rl_games_ppo_cfg.py file params
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HandGaitPPORunnerCfg",  # rsl_rl, oracle teacher
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HandGaitPPORunnerCNNCfg",  # rsl_rl, tactile image teacher
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

#Tactile Teacher#
gym.register(
    id="Hand-Gait-Tactile-v0",
    entry_point=f"{__name__}.hand_gait_env:HandGaitEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hand_gait_env_cfg:HandGaitEnvCfg",  # TODO: if have a separate cfg class python file, remember to change the path
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HandGaitPPORunnerCNNCfg",  # rsl_rl, tactile image teacher
    },
)

# Distillation Student #

gym.register(
    id="Hand-Gait-Distillation-v0",
    entry_point=f"{__name__}.hand_gait_distillation_env:HandGaitDistillationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hand_gait_env_cfg:HandGaitEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HandGaitDistillationRunnerCfg",
    },
)

gym.register(
    id="Hand-Gait-RNN-Distillation-v0",
    entry_point=f"{__name__}.hand_gait_distillation_env:HandGaitDistillationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hand_gait_env_cfg:HandGaitEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HandGaitDistillationRNNRunnerCfg",
    },
)

gym.register(
    id="Hand-Gait-CNN-RNN-Distillation-v0",
    entry_point=f"{__name__}.hand_gait_distillation_env:HandGaitDistillationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hand_gait_env_cfg:HandGaitEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HandGaitDistillationCNNRNNRunnerCfg",
    },
)
