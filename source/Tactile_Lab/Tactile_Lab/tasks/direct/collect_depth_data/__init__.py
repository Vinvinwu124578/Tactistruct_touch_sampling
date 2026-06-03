# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym


from . import agents

##
# Register Gym environments.
##

#Oracle Teacher#
gym.register(
    id="Collect-Depth-Data-v0",
    entry_point=f"{__name__}.collect_depth_data_env:CollectDepthDataEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.collect_depth_data_env_cfg:CollectDepthDataEnvCfg",  # TODO: if have a separate cfg class python file, remember to change the path
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CollectDepthDataPPORunnerCfg",  # rsl_rl, oracle teacher
    },
)
