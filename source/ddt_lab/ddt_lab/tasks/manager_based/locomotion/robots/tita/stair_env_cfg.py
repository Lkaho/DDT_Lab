"""Tita stair-climbing environment configuration for MlpEstimator and DreamWaQ."""

import copy
import math

import isaaclab.terrains as terrain_gen
import isaaclab.terrains.trimesh.mesh_terrains as mesh_terrains
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp

from .rough_env_cfg import CommandsCfg as RoughCommandsCfg
from .rough_env_cfg import EventCfg, RewardsCfg, TitaFeedforwardActionsCfg, TitaRoughEnvCfg, configure_forward_only_play_commands
from .rough_env_cfg import TerminationsCfg as RoughTerminationsCfg


def inverted_pyramid_stairs_width_curriculum_terrain(
    difficulty: float,
    cfg: "MeshInvertedPyramidStairsWidthCurriculumCfg",
):
    """Generate inverted stairs with step width decreasing as terrain difficulty increases."""
    terrain_cfg = copy.copy(cfg)
    min_difficulty_width, max_difficulty_width = cfg.step_width_range
    terrain_cfg.step_width = min_difficulty_width + difficulty * (max_difficulty_width - min_difficulty_width)
    return mesh_terrains.inverted_pyramid_stairs_terrain(difficulty, terrain_cfg)


@configclass
class MeshInvertedPyramidStairsWidthCurriculumCfg(terrain_gen.MeshInvertedPyramidStairsTerrainCfg):
    """Inverted pyramid stairs whose tread width is curriculum-scaled by terrain difficulty."""

    function = inverted_pyramid_stairs_width_curriculum_terrain

    step_width: float = 0.5
    """Fallback step width; overwritten from ``step_width_range`` during terrain generation."""

    step_width_range: tuple[float, float] = (0.5, 0.3)
    """Step width at difficulty 0 and difficulty 1."""


# STAIR_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
#     size=(8.0, 8.0),
#     border_width=20.0,
#     num_rows=10,
#     num_cols=10,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     slope_threshold=0.75,
#     difficulty_range=(0.0, 1.0),
#     use_cache=False,
#     curriculum=True,
#     sub_terrains={
#         "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
#             proportion=0.10,
#             noise_range=(0.01, 0.05),
#             noise_step=0.02,
#             border_width=0.25,
#         ),
#         "smooth_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
#             proportion=0.10,
#             slope_range=(0.0, 0.3),
#             platform_width=2.0,
#             border_width=0.25,
#         ),
#         "discrete_obstacles": terrain_gen.MeshRandomGridTerrainCfg(
#             proportion=0.20,
#             grid_width=0.45,
#             grid_height_range=(0.02, 0.10),
#             platform_width=2.0,
#         ),
#         "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
#             proportion=0.50,
#             step_height_range=(0.08, 0.15),
#             step_width=0.5,
#             platform_width=2.5,
#             border_width=0.0,
#             holes=False,
#         ),
#         "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
#     },
# )
STAIR_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "smooth_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.0, 0.3),
            platform_width=2.0,
            border_width=0.25,
        ),
        "discrete_obstacles": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.20,
            grid_width=0.45,
            grid_height_range=(0.0, 0.10),
            platform_width=2.0,
        ),
        "stairs_down": MeshInvertedPyramidStairsWidthCurriculumCfg(
            proportion=0.60,
            step_height_range=(0.0, 0.15),
            step_width_range=(0.5, 0.3),
            platform_width=2.5,
            border_width=0.0,
            holes=False,
        ),
    },
)


# STAIR_PLAY_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
#     size=(8.0, 8.0),
#     border_width=20.0,
#     num_rows=6,
#     num_cols=6,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     slope_threshold=0.75,
#     difficulty_range=(0.0, 1.0),
#     use_cache=False,
#     curriculum=False,
#     sub_terrains={
#         "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
#         "smooth_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
#             proportion=0.15,
#             slope_range=(0.0, 0.3),
#             platform_width=2.0,
#             border_width=0.25,
#         ),
#         "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
#             proportion=0.15,
#             noise_range=(0.01, 0.05),
#             noise_step=0.02,
#             border_width=0.25,
#         ),
#         "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
#             proportion=0.25,
#             step_height_range=(0.04, 0.1),
#             step_width=0.45,
#             platform_width=2.5,
#             border_width=0.0,
#             holes=False,
#         ),
#         "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
#             proportion=0.15,
#             step_height_range=(0.04, 0.1),
#             step_width=0.45,
#             platform_width=2.5,
#             border_width=0.0,
#             holes=False,
#         ),
#         "discrete_obstacles": terrain_gen.MeshRandomGridTerrainCfg(
#             proportion=0.15,
#             grid_width=0.45,
#             grid_height_range=(0.02, 0.10),
#             platform_width=2.0,
#         ),
#     },
# )

STAIR_PLAY_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=6,
    num_cols=6,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
        "smooth_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.3),
            platform_width=2.0,
            border_width=0.25,
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.10, 0.18),
            step_width=0.45,
            platform_width=2.5,
            border_width=0.0,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.10, 0.18),
            step_width=0.45,
            platform_width=2.5,
            border_width=0.0,
            holes=False,
        ),
    },
)


# Stair estimator settings are centralized here so future tuning only needs this file.
STAIR_ESTIMATOR_POLICY_HISTORY_LENGTH = 10
STAIR_ESTIMATOR_WINDOW_LENGTH = 3
STAIR_ESTIMATOR_OUTPUT_HISTORY_LENGTH = STAIR_ESTIMATOR_POLICY_HISTORY_LENGTH
STAIR_ESTIMATOR_HISTORY_TERM_DIMS = (3, 3, 3, 6, 8, 8)
STAIR_DREAMWAQ_HISTORY_LENGTH = 10


@configclass
class StairMlpEstimatorObservationsCfg:
    """Observation specification for stair climbing with a velocity estimator."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.diag_base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.diag_projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 0.0, 0.25),
        )
        joint_pos = ObsTerm(
            func=mdp.diag_joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_.*_leg_[123]"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.diag_joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )
        last_action = ObsTerm(func=mdp.diag_safe_blended_action, scale=1.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = STAIR_ESTIMATOR_POLICY_HISTORY_LENGTH

    @configclass
    class MlpEstimatorProprioHistoryCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = STAIR_ESTIMATOR_POLICY_HISTORY_LENGTH

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.diag_base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.diag_base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.diag_projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 0.0, 0.25),
        )
        joint_pos = ObsTerm(
            func=mdp.diag_joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_4"),
            },
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.diag_joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.diag_safe_blended_action, scale=1.0)
        feet_avg_contact_force = ObsTerm(
            func=mdp.diag_feet_average_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_leg_4"])},
            clip=(-1000.0, 1000.0),
            scale=0.01,
        )
        height_scan = ObsTerm(
            func=mdp.safe_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
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

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    @configclass
    class VelocityTargetCfg(ObsGroup):
        """Velocity supervision target for the estimator."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=1.0)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    history: MlpEstimatorProprioHistoryCfg = MlpEstimatorProprioHistoryCfg()
    critic: CriticCfg = CriticCfg()
    priv: PrivCfg = PrivCfg()
    velocity_target: VelocityTargetCfg = VelocityTargetCfg()


@configclass
class StairDreamWaQObservationsCfg:
    """DreamWaQ-style observations for stair climbing with context estimation."""

    @configclass
    class PolicyCfg(StairMlpEstimatorObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 1

    @configclass
    class DreamWaQProprioHistoryCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = STAIR_DREAMWAQ_HISTORY_LENGTH

    @configclass
    class CriticCfg(StairMlpEstimatorObservationsCfg.CriticCfg):
        robot_joint_torque = ObsTerm(
            func=mdp.robot_joint_torque,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        )
        robot_joint_acc = ObsTerm(
            func=mdp.robot_joint_acc,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        )
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_leg_4"])},
        )
        robot_mass = ObsTerm(
            func=mdp.robot_mass,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*")},
        )
        robot_inertia = ObsTerm(
            func=mdp.robot_inertia,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*")},
        )
        robot_joint_pos = ObsTerm(
            func=mdp.robot_joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        )
        robot_joint_stiffness = ObsTerm(
            func=mdp.robot_joint_stiffness,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        )
        robot_joint_damping = ObsTerm(
            func=mdp.robot_joint_damping,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        )
        robot_pos = ObsTerm(func=mdp.robot_pos)
        robot_vel = ObsTerm(func=mdp.robot_vel)
        robot_material_properties = ObsTerm(func=mdp.robot_material_properties)
        feet_contact_force = ObsTerm(
            func=mdp.feet_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_leg_4"])},
        )

    @configclass
    class PrivCfg(StairMlpEstimatorObservationsCfg.PrivCfg):
        pass

    @configclass
    class VelocityTargetCfg(ObsGroup):
        """Velocity supervision target for DreamWaQ; physical privileged terms live in ``priv``."""

        base_lin_vel = ObsTerm(func=mdp.diag_base_lin_vel, scale=1.0)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    history: DreamWaQProprioHistoryCfg = DreamWaQProprioHistoryCfg()
    critic: CriticCfg = CriticCfg()
    priv: PrivCfg = PrivCfg()
    velocity_target: VelocityTargetCfg = VelocityTargetCfg()


@configclass
class StairActionsCfg(TitaFeedforwardActionsCfg):
    """Action specification with contact-triggered feedforward trajectory."""
    pass


@configclass
class StairCommandsCfg(RoughCommandsCfg):
    """Commands for stair climbing.

    When an environment is assigned to the ``stairs_down`` terrain columns, we keep only the
    commanded forward velocity and disable lateral motion, while preserving heading control so
    the robot learns to stay aligned with the stair direction.
    """

    base_velocity = mdp.TerrainAwareUniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        align_heading_with_robot_on_reset=True,
        debug_vis=True,
        restricted_sub_terrain_names=("stairs_up", "stairs_down"),
        restricted_lin_vel_x_range=(0.0, 1.0),
        restricted_heading_range=None,
        force_zero_lin_vel_y=True,
        force_zero_ang_vel_z=False,
        disable_heading_command=False,
        ranges=mdp.TerrainAwareUniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class StairEventCfg(EventCfg):
    """Events for stair climbing."""


@configclass
class StairRewardsCfg(RewardsCfg):
    """Reward terms tailored for stair climbing."""

    # ---------------------------------------------------------------------
    # Task rewards
    # ---------------------------------------------------------------------
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    track_heading_exp = RewTerm(
        func=mdp.track_heading_exp,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        },
    )

    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_.*_leg_[123]"]),
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_leg_4"]),
            "threshold": 0.1,
            "triggered_only": True,
            "action_name": "joint_pos",
        },
    )

    feet_height = RewTerm(
        func=mdp.feet_height_band_relative,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_leg_4"]),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "target_height": 0.10,
            "std": 0.05,
            "tanh_mult": 2.0,
            "wheel_radius": 0.0925,
            "action_name": "joint_pos",
        },
    )

    feet_contact_number = RewTerm(
        func=mdp.feet_xy_swing_fz_stance_match,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_leg_4"]),
            "action_name": "joint_pos",
            "mismatch_penalty": 1.3,
            "swing_xy_threshold": 50.0,
            "stance_fz_threshold": 50.0,
        },
    )

    # feet_swing_xy_impact = RewTerm(
    #     func=mdp.feet_swing_xy_impact_penalty,
    #     weight=-0.002,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_leg_4"]),
    #         "action_name": "joint_pos",
    #         "xy_force_threshold": 50.0,
    #     },
    # )

    tracking_target_pos = RewTerm(
        func=mdp.track_ff_target_pos_exp,
        weight=1.0,
        params={
            "action_name": "joint_pos",
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.1,
        },
    )

    # ---------------------------------------------------------------------
    # Style rewards
    # ---------------------------------------------------------------------

    # joint_mirror = RewTerm(
    #     func=mdp.stair_joint_mirror,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "mirror_joints": [["joint_left_leg_(1|2|3)", "joint_right_leg_(1|2|3)"]],
    #         "action_name": "joint_pos",
    #     },
    # )
    joint_mirror = None

    joint_deviation_leg1_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_.*_leg_1"])},
    )

    wheel_vel_penalty = RewTerm(
        func=mdp.wheel_zero_velocity_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["joint_left_leg_4", "joint_right_leg_4"],
                preserve_order=True,
            ),
            "action_name": "joint_pos",
        },
    )

    wheel_spin = RewTerm(
        func=mdp.wheel_spin_penalty,
        weight=-5.0,
        params={
            "wheel_joint_cfg": SceneEntityCfg(
                "robot", joint_names=["joint_left_leg_4", "joint_right_leg_4"]
            ),
            "foot_body_cfg": SceneEntityCfg("robot", body_names=["left_leg_4", "right_leg_4"]),
            "wheel_radius": 0.0925,
            "spin_scale": 0.8,
            "slip_deadband": 0.1,
        },
    )

    feet_y_distance = RewTerm(
        func=mdp.feet_y_distance,
        weight=-2.0,
        params={
            "min_distance": 0.5,
            "max_distance": 0.62,
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_leg_4"]),
        },
    )

    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-30.0,
        params={"target_height": 0.38, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-12.0)

    upward = RewTerm(func=mdp.upward, weight= 0.5)

    # ---------------------------------------------------------------------
    # Regularization rewards
    # ---------------------------------------------------------------------
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_safe, weight=-0.01, params={"clamp_value": 5.0})
    action_smooth = RewTerm(
        func=mdp.action_smooth_for_term_indices_safe,
        weight=-0.01,
        params={
            "action_name": "joint_pos",
            "action_indices": [3, 7],
            "clamp_value": 5.0,
            "scale_with_term": False,
        },
    )

    zero_command_base_motion = RewTerm(
        func=mdp.zero_command_base_motion_l1,
        weight=-1.0,
        params={
            "lin_threshold": 0.05,
            "ang_threshold": 0.05,
        },
    )
    opposite_base_vel = RewTerm(
        func=mdp.opposite_base_vel,
        weight=-40.0,
        params={"command_name": "base_velocity"},
    )

    opposite_wheel_vel = RewTerm(
        func=mdp.opposite_wheel_vel,
        weight=-1.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_4"]),
        },
    )

@configclass
class StairCurriculumCfg:
    """Curriculum for stair climbing.

    We keep terrain curriculum from the rough task.

    Note:
        The k_ff linear annealing is executed inside the feedforward action term on every
        environment step, so it stays synchronized with how feedforward is actually injected.
    """

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    terrain_level = CurrTerm(
        func=mdp.terrain_levels_by_type,
        params={
            "terrain_names": [
                "smooth_slope",
                "discrete_obstacles",
                "stairs_down",
            ],
        },
    )


@configclass
class StairTerminationsCfg(RoughTerminationsCfg):
    """Termination terms for stair climbing."""

    illegal_leg2_contact = None
    illegal_leg3_contact = None
    abnormal_blended_action = DoneTerm(func=mdp.abnormal_blended_action_termination, params={"threshold": 100.0})


@configclass
class TitaStairBaseEnvCfg(TitaRoughEnvCfg):
    """Common stair-climbing environment configuration."""

    commands: StairCommandsCfg = StairCommandsCfg()
    actions: StairActionsCfg = StairActionsCfg()
    rewards: StairRewardsCfg = StairRewardsCfg()
    terminations: StairTerminationsCfg = StairTerminationsCfg()
    events: StairEventCfg = StairEventCfg()
    curriculum: StairCurriculumCfg = StairCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIR_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 2

        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.align_heading_with_robot_on_reset = True
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (-math.pi / 4.0, math.pi / 4.0)
        self.commands.base_velocity.restricted_lin_vel_x_range = (0.0, 1.0)
        self.commands.base_velocity.restricted_heading_range = None

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-math.pi / 4.0, math.pi / 4.0)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
        }
      
        self.rewards.base_height_l2.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        


@configclass
class TitaStairMlpEstimatorEnvCfg(TitaStairBaseEnvCfg):
    """Tita stair-climbing environment with MlpEstimator velocity estimation."""

    observations: StairMlpEstimatorObservationsCfg = StairMlpEstimatorObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.only_positive_rewards = False


@configclass
class TitaStairDreamWaQEnvCfg(TitaStairBaseEnvCfg):
    """Tita stair-climbing environment with DreamWaQ context estimation for AdaBoot."""

    observations: StairDreamWaQObservationsCfg = StairDreamWaQObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.only_positive_rewards = False
        self.rewards.base_height_l2.weight = -30.0
        self.rewards.flat_orientation_l2.weight = -12.0
        self.rewards.opposite_base_vel.weight = -40.0
        self.rewards.opposite_wheel_vel.weight = -1.5
        self.rewards.feet_y_distance.weight = -2.0


@configclass
class TitaStairMlpEstimatorEnvCfg_PLAY(TitaStairMlpEstimatorEnvCfg):
    """Play configuration for the stair MlpEstimator environment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        configure_forward_only_play_commands(self)

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIR_PLAY_TERRAINS_CFG
        self.observations.policy.enable_corruption = False
        self.observations.history.enable_corruption = False

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None


@configclass
class TitaStairDreamWaQEnvCfg_PLAY(TitaStairDreamWaQEnvCfg):
    """Play configuration for the stair DreamWaQ environment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        configure_forward_only_play_commands(self)

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIR_PLAY_TERRAINS_CFG
        self.observations.policy.enable_corruption = False
        self.observations.history.enable_corruption = False

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None
