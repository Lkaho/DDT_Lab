# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""D1H stair-climbing environment configuration for DreamWaQ."""

import copy
import math

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import isaaclab.terrains.trimesh.mesh_terrains as mesh_terrains
from ddt_lab.assets.ddt_robot import DDT_D1H_CFG
from ddt_lab.managers import CostTermCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

D1H_LEG_JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
]
D1H_WHEEL_JOINT_NAMES = ["FL_foot_joint", "FR_foot_joint"]
D1H_LEG_JOINT_REGEX = "F[LR]_(hip|thigh|calf)_joint"
D1H_WHEEL_JOINT_REGEX = "F[LR]_foot_joint"
D1H_FOOT_BODY_REGEX = "F[LR]_foot"
D1H_WHEEL_RADIUS = 0.087
D1H_STAIR_DREAMWAQ_HISTORY_LENGTH = 10
D1H_BASE_HEIGHT_TARGET = 0.45
D1H_DEFAULT_FEET_Y_DISTANCE = 0.4406243
D1H_FEET_Y_DISTANCE_STD = 0.05
D1H_FEET_Y_DISTANCE_LIMIT_RANGE = (0.40, 0.50)


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
    step_width_range: tuple[float, float] = (0.5, 0.3)


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


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the stair terrain scene with D1H."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAIR_TERRAINS_CFG,
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = DDT_D1H_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 5.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


@configclass
class CommandsCfg:
    """Command specifications for stair climbing."""

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
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi / 4.0, math.pi / 4.0),
        ),
    )


@configclass
class ActionsCfg:
    """D1H stair action specification with contact-triggered feedforward."""

    joint_pos = mdp.D1HJointPositionEffortActionCfg(
        asset_name="robot",
        leg_joint_names=D1H_LEG_JOINT_NAMES,
        wheel_joint_names=D1H_WHEEL_JOINT_NAMES,
        leg_scale=(0.25, 0.25, 0.25, 0.25, 0.25, 0.25),
        wheel_scale=5.0,
        wheel_effort_gain=0.0,
        wheel_offset=0.0,
        use_default_leg_offset=True,
        preserve_order=True,
        clip={".*": (-100.0, 100.0)},
        command_delay_min_steps=0,
        command_delay_max_steps=1,
        feedforward_enabled=True,
        k_fb=1.0,
        k_ff=0.3,
        feedforward_period=0.6,
        feedforward_amplitude={
            "F[LR]_thigh_joint": 0.4,
            "F[LR]_calf_joint": -0.80,
        },
        feedforward_joint_names=[
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
        ],
        contact_trigger_enabled=True,
        contact_sensor_name="contact_forces",
        contact_body_pattern=D1H_FOOT_BODY_REGEX,
        contact_force_threshold=50.0,
        contact_trigger_warmup_steps=15,
        inter_leg_phase_lag=0.8,
        k_ff_anneal_enabled=True,
        k_ff_final=0.0,
        k_ff_start_iteration=6000,
        k_ff_anneal_iterations=6000,
        k_ff_steps_per_iteration=24,
    )


@configclass
class D1HStairDreamWaQObservationsCfg:
    """DreamWaQ observation groups for D1H stair climbing."""

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
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[D1H_LEG_JOINT_REGEX])},
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

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1

    @configclass
    class DreamWaQProprioHistoryCfg(PolicyCfg):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.history_length = D1H_STAIR_DREAMWAQ_HISTORY_LENGTH

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
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=D1H_WHEEL_JOINT_REGEX),
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
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[D1H_FOOT_BODY_REGEX])},
            clip=(-1000.0, 1000.0),
            scale=0.01,
        )
        height_scan = ObsTerm(
            func=mdp.safe_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
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
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[D1H_FOOT_BODY_REGEX])},
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
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[D1H_FOOT_BODY_REGEX])},
        )

        def __post_init__(self) -> None:
            self.history_length = 1

    @configclass
    class PrivCfg(ObsGroup):
        contact_state = ObsTerm(
            func=mdp.contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[D1H_FOOT_BODY_REGEX])},
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
        base_lin_vel = ObsTerm(func=mdp.diag_base_lin_vel, scale=1.0)

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
class EventCfg:
    """Configuration for D1H stair domain randomization events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.6),
            "dynamic_friction_range": (0.2, 1.6),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.5, 2.0),
            "operation": "add",
        },
    )
    add_base_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "inertia_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    add_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-10.0, 10.0),
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-math.pi / 4.0, math.pi / 4.0)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (-0.5, 1.0), "velocity_range": (-0.0, 0.0)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for D1H stair climbing."""

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
        weight=0.8,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-10.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
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
    # stand_still = RewTerm(
    #     func=mdp.stand_still,
    #     weight=-1.0,
    #     params={
    #         "command_name": "base_velocity",
    #         "command_threshold": 0.1,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[D1H_LEG_JOINT_REGEX]),
    #     },
    # )
    joint_mirror = None
    joint_deviation_hips_l1 = None
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[D1H_FOOT_BODY_REGEX]),
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
            "asset_cfg": SceneEntityCfg("robot", body_names=[D1H_FOOT_BODY_REGEX]),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "target_height": 0.10,
            "std": 0.05,
            "tanh_mult": 2.0,
            "wheel_radius": D1H_WHEEL_RADIUS,
            "action_name": "joint_pos",
        },
    )
    feet_contact_number = RewTerm(
        func=mdp.feet_xy_swing_fz_stance_match,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[D1H_FOOT_BODY_REGEX]),
            "action_name": "joint_pos",
            "mismatch_penalty": 1.3,
            "swing_xy_threshold": 50.0,
            "stance_fz_threshold": 50.0,
        },
    )
    tracking_target_pos = RewTerm(
        func=mdp.track_ff_target_pos_exp,
        weight=1.0,
        params={
            "action_name": "joint_pos",
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.1,
        },
    )
    # wheel_vel_penalty = RewTerm(
    #     func=mdp.wheel_swing_velocity_l2,
    #     weight=-0.2,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=D1H_WHEEL_JOINT_NAMES, preserve_order=True),
    #         "action_name": "joint_pos",
    #     },
    # )
    wheel_spin = RewTerm(
        func=mdp.wheel_spin_penalty,
        weight=-5.0,
        params={
            "wheel_joint_cfg": SceneEntityCfg("robot", joint_names=D1H_WHEEL_JOINT_NAMES),
            "foot_body_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot"]),
            "wheel_radius": D1H_WHEEL_RADIUS,
            "spin_scale": 0.8,
            "slip_deadband": 0.1,
        },
    )
    # wheel_body_speed_mismatch = RewTerm(
    #     func=mdp.wheel_body_speed_mismatch_l1,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=D1H_WHEEL_JOINT_NAMES, preserve_order=True),
    #         "wheel_radius": D1H_WHEEL_RADIUS,
    #     },
    # )
    zero_command_wheel_vel = RewTerm(
        func=mdp.zero_command_wheel_vel_l1,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[D1H_WHEEL_JOINT_REGEX]),
        },
    )

    opposite_base_vel = RewTerm(
        func=mdp.opposite_base_vel,
        weight=-5.0,
        params={"command_name": "base_velocity"},
    )
    opposite_wheel_vel = RewTerm(
        func=mdp.opposite_wheel_vel,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[D1H_WHEEL_JOINT_REGEX]),
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["F[LR]_(hip|thigh|calf)"]), "threshold": 1.0},
    )
    upright = RewTerm(func=mdp.upright, weight=0.25)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-12.0)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-15.0,
        params={"target_height": D1H_BASE_HEIGHT_TARGET, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    feet_y_distance = RewTerm(
        func=mdp.feet_y_distance_exp,
        weight=1.0,
        params={
            "target_distance": D1H_DEFAULT_FEET_Y_DISTANCE,
            "std": D1H_FEET_Y_DISTANCE_STD,
            "asset_cfg": SceneEntityCfg("robot", body_names=[D1H_FOOT_BODY_REGEX]),
        },
    )
    feet_y_distance_limit = RewTerm(
        func=mdp.feet_y_distance,
        weight=-20.0,
        params={
            "min_distance": D1H_FEET_Y_DISTANCE_LIMIT_RANGE[0],
            "max_distance": D1H_FEET_Y_DISTANCE_LIMIT_RANGE[1],
            "asset_cfg": SceneEntityCfg("robot", body_names=[D1H_FOOT_BODY_REGEX]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for D1H stair climbing."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )
    abnormal_blended_action = DoneTerm(func=mdp.abnormal_blended_action_termination, params={"threshold": 100.0})


@configclass
class CurriculumCfg:
    """Curriculum terms for D1H stair climbing."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    terrain_level = CurrTerm(
        func=mdp.terrain_levels_by_type,
        params={"terrain_names": ["smooth_slope", "discrete_obstacles", "stairs_down"]},
    )


@configclass
class CostsCfg:
    """Constraint costs for D1H stair climbing."""

    joint_pos_limit = CostTermCfg(
        func=mdp.joint_pos_limit,
        scale=1.0,
        d_value=0.0,
        k_value=0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[D1H_LEG_JOINT_REGEX])},
    )
    joint_vel_limit = CostTermCfg(
        func=mdp.joint_vel_limit,
        scale=1.0,
        d_value=0.0,
        k_value=0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_torque_limit = CostTermCfg(
        func=mdp.joint_torque_limit,
        scale=1.0,
        d_value=0.0,
        k_value=0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


def configure_forward_only_play_commands(env_cfg, lin_vel_x_range: tuple[float, float] = (0.0, 1.0)) -> None:
    """Configure play-mode velocity commands to sample only forward motion."""
    base_velocity = env_cfg.commands.base_velocity
    base_velocity.rel_standing_envs = 0.0
    base_velocity.ranges.lin_vel_x = lin_vel_x_range
    base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    if hasattr(base_velocity, "restricted_lin_vel_x_range"):
        base_velocity.restricted_lin_vel_x_range = lin_vel_x_range


@configclass
class D1HStairDreamWaQEnvCfg(ManagerBasedRLEnvCfg):
    """D1H stair terrain environment with DreamWaQ context estimation."""

    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: D1HStairDreamWaQObservationsCfg = D1HStairDreamWaQObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    costs: CostsCfg = CostsCfg()
    only_positive_rewards: bool = False

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        elif self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class D1HStairDreamWaQEnvCfg_PLAY(D1HStairDreamWaQEnvCfg):
    """Play configuration for the D1H stair DreamWaQ environment."""

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
