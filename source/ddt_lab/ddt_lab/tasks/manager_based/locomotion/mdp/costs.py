# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Non-negative cost terms for constrained locomotion training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Sum of joint-position excursions outside soft joint limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    soft_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    lower_violation = -(joint_pos - soft_limits[..., 0]).clamp(max=0.0)
    upper_violation = (joint_pos - soft_limits[..., 1]).clamp(min=0.0)
    return torch.sum(lower_violation + upper_violation, dim=1)


def joint_torque_limit(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_ratio: float = 0.9,
) -> torch.Tensor:
    """Sum of absolute applied torque above ``soft_ratio * effort_limit``."""
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    limit = asset.data.joint_effort_limits[:, asset_cfg.joint_ids] * soft_ratio
    return torch.sum((torque.abs() - limit).clamp(min=0.0), dim=1)


def joint_vel_limit(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_ratio: float = 0.9,
) -> torch.Tensor:
    """Sum of joint velocity above ``soft_ratio * velocity_limit``, capped per joint."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    limit = asset.data.joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
    return torch.sum((joint_vel.abs() - limit).clamp(min=0.0, max=1.0), dim=1)
