# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp

from .rough_env_cfg import TitaRoughEnvCfg, configure_forward_only_play_commands

ESTIMATOR_TARGET_BASE_LIN_VEL_XY_SCALE = [1.0, 1.0]
ESTIMATOR_POLICY_BASE_LIN_VEL_XY_SCALE = [2.0, 2.0]
ESTIMATOR_HISTORY_LENGTH = 3
ESTIMATOR_FEATURE_HISTORY_LENGTH = 10


FLAT_DREAMWAQ_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.5,
            noise_range=(0.01, 0.05),
            noise_step=0.02,
            border_width=0.25,
        ),
    },
)


@configclass
class MlpEstimatorObservationsCfg:
    """Observation specification for MlpEstimator without policy-side base linear velocity."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 0.0, 0.25),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_.*_leg_[123]"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )
        last_action = ObsTerm(func=mdp.last_action, scale=1.0)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = ESTIMATOR_FEATURE_HISTORY_LENGTH

    @configclass
    class MlpEstimatorProprioHistoryCfg(PolicyCfg):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.history_length = ESTIMATOR_FEATURE_HISTORY_LENGTH

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 0.0, 0.25),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_4"),
            },
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, scale=1.0)
        height_scan = ObsTerm(
            func=mdp.safe_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self) -> None:
            self.history_length = 1

    @configclass
    class PrivCfg(ObsGroup):
        """Privileged physical terms for the critic only."""

        contact_state = ObsTerm(
            func=mdp.contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_leg_4"])},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        joint_kp_factor = ObsTerm(
            func=mdp.joint_kp_factor,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(0.0, 2.0),
            scale=1.0,
        )
        joint_kd_factor = ObsTerm(
            func=mdp.joint_kd_factor,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(0.0, 2.0),
            scale=1.0,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    @configclass
    class VelocityTargetCfg(ObsGroup):
        base_lin_vel_xy = ObsTerm(func=mdp.base_lin_vel_xy, scale=1.0)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    history: MlpEstimatorProprioHistoryCfg = MlpEstimatorProprioHistoryCfg()
    critic: CriticCfg = CriticCfg()
    priv: PrivCfg = PrivCfg()
    velocity_target: VelocityTargetCfg = VelocityTargetCfg()


@configclass
class DreamWaQObservationsCfg:
    """Rough-style observations for flat/rough-flat DreamWaQ training."""

    @configclass
    class PolicyCfg(MlpEstimatorObservationsCfg.PolicyCfg):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1

    @configclass
    class DreamWaQProprioHistoryCfg(PolicyCfg):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.history_length = 5

    @configclass
    class CriticCfg(MlpEstimatorObservationsCfg.CriticCfg):
        feet_avg_contact_force = ObsTerm(
            func=mdp.diag_feet_average_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_leg_4"])},
            clip=(-1000.0, 1000.0),
            scale=0.01,
        )

    @configclass
    class PrivCfg(MlpEstimatorObservationsCfg.PrivCfg):
        pass

    @configclass
    class VelocityTargetCfg(ObsGroup):
        """Velocity supervision target for DreamWaQ; physical privileged terms live in ``priv``."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=1.0)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    history: DreamWaQProprioHistoryCfg = DreamWaQProprioHistoryCfg()
    critic: CriticCfg = CriticCfg()
    priv: PrivCfg = PrivCfg()
    velocity_target: VelocityTargetCfg = VelocityTargetCfg()


@configclass
class TitaRoughMlpEstimatorEnvCfg(TitaRoughEnvCfg):
    """Tita rough terrain environment with history-based base velocity estimation."""

    observations: MlpEstimatorObservationsCfg = MlpEstimatorObservationsCfg()


@configclass
class TitaRoughMlpEstimatorEnvCfg_PLAY(TitaRoughMlpEstimatorEnvCfg):
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


@configclass
class TitaFlatMlpEstimatorEnvCfg(TitaRoughEnvCfg):
    """Tita flat terrain environment with history-based base velocity estimation."""

    observations: MlpEstimatorObservationsCfg = MlpEstimatorObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        self.commands.base_velocity.rel_standing_envs = 0.1

        self.rewards.feet_y_distance = RewTerm(
            func=mdp.feet_y_distance,
            weight=-2.0,
            params={
                "min_distance": 0.5,
                "max_distance": 0.6,
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*_leg_4"]),
            },
        )
        self.rewards.stand_still = RewTerm(
            func=mdp.stand_still,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.15,
                "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_.*_leg_[123]"]),
            },
        )
        self.rewards.zero_command_wheel_vel = RewTerm(
            func=mdp.zero_command_wheel_vel_l1,
            weight=-0.1,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.15,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_4"]),
            },
        )
        self.rewards.opposite_wheel_vel = RewTerm(
            func=mdp.opposite_wheel_vel,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_4"]),
            },
        )


@configclass
class TitaFlatMlpEstimatorEnvCfg_PLAY(TitaFlatMlpEstimatorEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        configure_forward_only_play_commands(self)

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.history.enable_corruption = False

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None


@configclass
class TitaRoughDreamWaQEnvCfg(TitaRoughEnvCfg):
    """Tita rough terrain environment with DreamWaQ context estimation."""

    observations: DreamWaQObservationsCfg = DreamWaQObservationsCfg()


@configclass
class TitaRoughDreamWaQEnvCfg_PLAY(TitaRoughDreamWaQEnvCfg):
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


@configclass
class TitaFlatDreamWaQEnvCfg(TitaRoughEnvCfg):
    """Tita flat/rough-flat environment with DreamWaQ context estimation."""

    observations: DreamWaQObservationsCfg = DreamWaQObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = FLAT_DREAMWAQ_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 1

        self.only_positive_rewards = True
        self.rewards.flat_orientation_l2.weight = -12.0
        self.rewards.base_height_l2.weight = -5.0
        self.rewards.joint_deviation_legs_l1.weight = -0.5
        self.rewards.undesired_contacts.weight = -5.0


@configclass
class TitaFlatDreamWaQEnvCfg_PLAY(TitaFlatDreamWaQEnvCfg):
    """Play configuration for the flat DreamWaQ environment."""

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
