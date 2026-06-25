# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

FLAT_ROUGH_DREAMWAQ_HISTORY_LENGTH = 10
D1H_STAIR_DREAMWAQ_HISTORY_LENGTH = 10

D1H_COST_NAMES = ["joint_pos_limit", "joint_vel_limit", "joint_torque_limit"]
D1H_COST_K_VALUES = [0.01, 0.01, 0.01]
D1H_COST_D_VALUES = [0.0, 0.0, 0.0]


@configclass
class D1HRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "d1h_rough"
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


def _enable_np3o_constraints(runner_cfg) -> None:
    runner_cfg.policy.num_costs = len(D1H_COST_NAMES)
    runner_cfg.algorithm.num_costs = len(D1H_COST_NAMES)
    runner_cfg.algorithm.cost_names = list(D1H_COST_NAMES)
    runner_cfg.algorithm.cost_k_values = list(D1H_COST_K_VALUES)
    runner_cfg.algorithm.cost_d_values = list(D1H_COST_D_VALUES)
    runner_cfg.algorithm.cost_k_growth = 1.0004
    runner_cfg.algorithm.cost_k_max = 1.0
    runner_cfg.algorithm.cost_value_loss_coef = 1.0
    runner_cfg.algorithm.cost_viol_loss_coef = 1.0


@configclass
class D1HDreamWaQAdaBootPPORunnerCfg(D1HRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 30000
        self.experiment_name = "d1h_dreamwaq_adaboot"
        self.empirical_normalization = None
        self.policy.class_name = "ActorCriticWithCENet"
        self.algorithm.class_name = "PPOWithCENetAdaBoot"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
        self.policy.cenet_encoder_hidden_dims = [128, 64]
        self.policy.cenet_decoder_hidden_dims = [64, 128]
        self.policy.cenet_velocity_dim = 3
        self.policy.cenet_latent_dim = 16
        self.policy.num_history = FLAT_ROUGH_DREAMWAQ_HISTORY_LENGTH
        self.algorithm.cenet_loss_coef = 1.0
        self.algorithm.cenet_velocity_loss_coef = 1.0
        self.algorithm.cenet_reconstruction_loss_coef = 1.0
        self.algorithm.cenet_kl_loss_coef = 0.01
        self.algorithm.vae_learning_rate = 1.0e-3
        self.algorithm.num_vae_substeps = 1
        self.algorithm.rl_grad_to_cenet = True
        self.algorithm.adaboot_enabled = True
        self.algorithm.adaboot_reward_window = 128
        self.algorithm.adaboot_min_episodes = 32
        _enable_np3o_constraints(self)
        self.obs_groups = {
            "policy": ["policy"],
            "critic": ["critic", "priv"],
            "history": ["history"],
            "velocity_target": ["velocity_target"],
        }


@configclass
class D1HFlatDreamWaQAdaBootPPORunnerCfg(D1HDreamWaQAdaBootPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "d1h_flat_dreamwaq_adaboot"


@configclass
class D1HRoughDreamWaQAdaBootPPORunnerCfg(D1HDreamWaQAdaBootPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 5000
        self.experiment_name = "d1h_rough_dreamwaq_adaboot"


@configclass
class D1HStairDreamWaQAdaBootPPORunnerCfg(D1HRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 30000
        self.experiment_name = "d1h_stair_dreamwaq_adaboot"
        self.empirical_normalization = None
        self.policy.class_name = "ActorCriticWithCENet"
        self.algorithm.class_name = "PPOWithCENetAdaBoot"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
        self.policy.cenet_encoder_hidden_dims = [128, 64]
        self.policy.cenet_decoder_hidden_dims = [64, 128]
        self.policy.cenet_velocity_dim = 3
        self.policy.cenet_latent_dim = 16
        self.policy.num_history = D1H_STAIR_DREAMWAQ_HISTORY_LENGTH
        self.algorithm.cenet_loss_coef = 1.0
        self.algorithm.cenet_velocity_loss_coef = 1.0
        self.algorithm.cenet_reconstruction_loss_coef = 1.0
        self.algorithm.cenet_kl_loss_coef = 0.01
        self.algorithm.vae_learning_rate = 1.0e-3
        self.algorithm.num_vae_substeps = 1
        self.algorithm.rl_grad_to_cenet = True
        self.algorithm.adaboot_enabled = True
        self.algorithm.adaboot_reward_window = 128
        self.algorithm.adaboot_min_episodes = 32
        _enable_np3o_constraints(self)
        self.obs_groups = {
            "policy": ["policy"],
            "critic": ["critic", "priv"],
            "history": ["history"],
            "velocity_target": ["velocity_target"],
        }
