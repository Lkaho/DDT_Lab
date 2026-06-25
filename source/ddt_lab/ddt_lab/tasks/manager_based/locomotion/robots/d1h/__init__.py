# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg, stair_env_cfg

##
# Register Gym environments.
##

POSITIVE_REWARD_ENV_ENTRY_POINT = "ddt_lab.tasks.manager_based.locomotion.positive_reward_env:PositiveRewardManagerBasedRLEnv"


gym.register(
    id="DDT-Velocity-Flat-D1H-DreamWaQ-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.D1HFlatDreamWaQEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:D1HFlatDreamWaQAdaBootPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Flat-D1H-DreamWaQ-Play-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.D1HFlatDreamWaQEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:D1HFlatDreamWaQAdaBootPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-D1H-DreamWaQ-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.D1HRoughDreamWaQEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:D1HRoughDreamWaQAdaBootPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-D1H-DreamWaQ-Play-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.D1HRoughDreamWaQEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:D1HRoughDreamWaQAdaBootPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Stair-D1H-DreamWaQ-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stair_env_cfg.D1HStairDreamWaQEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:D1HStairDreamWaQAdaBootPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Stair-D1H-DreamWaQ-Play-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stair_env_cfg.D1HStairDreamWaQEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:D1HStairDreamWaQAdaBootPPORunnerCfg",
    },
)
