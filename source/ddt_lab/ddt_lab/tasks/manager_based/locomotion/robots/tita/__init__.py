# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, stair_env_cfg

##
# Register Gym environments.
##

POSITIVE_REWARD_ENV_ENTRY_POINT = "ddt_lab.tasks.manager_based.locomotion.positive_reward_env:PositiveRewardManagerBasedRLEnv"


# MlpEstimator tasks.
gym.register(
    id="DDT-Velocity-Flat-Tita-MlpEstimator-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TitaFlatMlpEstimatorEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaFlatMlpEstimatorPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Flat-Tita-MlpEstimator-Play-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TitaFlatMlpEstimatorEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaFlatMlpEstimatorPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-Tita-MlpEstimator-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TitaRoughMlpEstimatorEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaRoughMlpEstimatorPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-Tita-MlpEstimator-Play-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TitaRoughMlpEstimatorEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaRoughMlpEstimatorPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Stair-Tita-MlpEstimator-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stair_env_cfg.TitaStairMlpEstimatorEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaStairMlpEstimatorPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Stair-Tita-MlpEstimator-Play-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stair_env_cfg.TitaStairMlpEstimatorEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaStairMlpEstimatorPPORunnerCfg",
    },
)


# DreamWaQ tasks.
gym.register(
    id="DDT-Velocity-Flat-Tita-DreamWaQ-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TitaFlatDreamWaQEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaFlatDreamWaQAdaBootPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Flat-Tita-DreamWaQ-Play-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TitaFlatDreamWaQEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaFlatDreamWaQAdaBootPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-Tita-DreamWaQ-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TitaRoughDreamWaQEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaRoughDreamWaQAdaBootPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-Tita-DreamWaQ-Play-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TitaRoughDreamWaQEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaRoughDreamWaQAdaBootPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Stair-Tita-DreamWaQ-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stair_env_cfg.TitaStairDreamWaQEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaStairDreamWaQAdaBootPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Stair-Tita-DreamWaQ-Play-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stair_env_cfg.TitaStairDreamWaQEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaStairDreamWaQAdaBootPPORunnerCfg",
    },
)
