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

#Oracle Teacher#
gym.register(
    id="BiPush-Oracle-v0",
    entry_point=f"{__name__}.bipush_env:BiPushEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bipush_env_cfg:BiPushEnvCfg",  # TODO: if have a separate cfg class python file, remember to change the path
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",  # rl_games, for both oracle and tactile, but need to modify the rl_games_ppo_cfg.py file params
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BipushPPORunnerCfg",  # rsl_rl, oracle teacher
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BipushPPORunnerCNNCfg",  # rsl_rl, tactile image teacher
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


#Tactile Teacher#
gym.register(
    id="BiPush-Tactile-v0",
    entry_point=f"{__name__}.bipush_env:BiPushEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bipush_env_cfg:BiPushEnvCfg",  # TODO: if have a separate cfg class python file, remember to change the path
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",  # rl_games, for both oracle and tactile, but need to modify the rl_games_ppo_cfg.py file params
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BipushPPORunnerCfg",  # rsl_rl, oracle teacher
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BipushPPORunnerCNNCfg",  # rsl_rl, tactile image teacher
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
