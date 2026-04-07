# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class TitaRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "tita_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class TitaFlatPPORunnerCfg(TitaRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "tita_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]


def _enable_velocity_estimator(runner_cfg, experiment_name: str, num_history: int = 3) -> None:
    runner_cfg.experiment_name = experiment_name
    runner_cfg.policy.class_name = "ActorCriticWithEstimator"
    runner_cfg.algorithm.class_name = "PPOWithEstimator"
    runner_cfg.policy.estimator_hidden_dims = [256, 128]
    runner_cfg.policy.num_history = num_history
    runner_cfg.policy.estimator_output_dim = 2
    runner_cfg.algorithm.estimator_loss_coef = 1.0
    runner_cfg.obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "history": ["history"],
        "privileged": ["privileged"],
    }


@configclass
class TitaRoughNoBaseVelEstimatorPPORunnerCfg(TitaRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        _enable_velocity_estimator(self, "tita_rough_no_base_vel_estimator", num_history=10)
        self.policy.estimated_history_length = 10
        self.policy.history_term_dims = [3, 3, 3, 6, 8, 8]
        self.policy.deploy_share_policy_and_history = True


@configclass
class TitaFlatNoBaseVelEstimatorPPORunnerCfg(TitaFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        _enable_velocity_estimator(self, "tita_flat_no_base_vel_estimator", num_history=10)
        self.policy.estimated_history_length = 10
        self.policy.history_term_dims = [3, 3, 3, 6, 8, 8]
        self.policy.deploy_share_policy_and_history = True


@configclass
class TitaStairEstimatorPPORunnerCfg(TitaRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 30000
        self.empirical_normalization = None
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
        _enable_velocity_estimator(self, "tita_stair_estimator", num_history=5)
        self.algorithm.estimator_loss_coef = 0.2


@configclass
class TitaStairPPORunnerCfg(TitaRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 30000
        self.experiment_name = "tita_stair"
        self.empirical_normalization = None
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
