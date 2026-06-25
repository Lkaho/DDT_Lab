# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import D1HFeedforwardActionsCfg, D1HDreamWaQObservationsCfg, D1HRoughEnvCfg
from .rough_env_cfg import configure_forward_only_play_commands


@configclass
class D1HFlatDreamWaQEnvCfg(D1HRoughEnvCfg):
    """D1H flat/rough-flat environment with DreamWaQ context estimation."""

    actions: D1HFeedforwardActionsCfg = D1HFeedforwardActionsCfg()
    observations: D1HDreamWaQObservationsCfg = D1HDreamWaQObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None

        self.only_positive_rewards = False
        # self.rewards.flat_orientation_l2.weight = -12.0
        # self.rewards.base_height_l2.weight = -20.0
        # self.rewards.joint_deviation_legs_l1.weight = -0.1
        # self.rewards.undesired_contacts.weight = -5.0


@configclass
class D1HFlatDreamWaQEnvCfg_PLAY(D1HFlatDreamWaQEnvCfg):
    """Play configuration for the D1H flat DreamWaQ environment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        configure_forward_only_play_commands(self)

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.observations.policy.enable_corruption = False
        self.observations.history.enable_corruption = False

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None
