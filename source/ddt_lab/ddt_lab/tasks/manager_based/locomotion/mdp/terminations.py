# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bad_base_height(
    env: ManagerBasedRLEnv,
    min_height: float = 0.22,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Terminate when the base is too low.

    If a height scanner is provided, ``min_height`` is interpreted relative to the
    terrain below the base. Without a scanner, it is interpreted as a world-frame
    z-height threshold.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_pos_w = asset.data.root_pos_w
    base_z = base_pos_w[:, 2]

    if sensor_cfg is None:
        return base_z < min_height

    sensor: RayCaster = env.scene[sensor_cfg.name]
    ray_hits = sensor.data.ray_hits_w
    ray_xy = ray_hits[..., :2]
    ray_z = ray_hits[..., 2]

    valid_hits = torch.isfinite(ray_z)
    has_valid_hit = valid_hits.any(dim=1)
    distances = torch.sum((base_pos_w[:, None, :2] - ray_xy) ** 2, dim=-1)
    distances = torch.where(valid_hits, distances, torch.full_like(distances, float("inf")))

    nearest_ids = torch.argmin(distances, dim=1)
    ground_z = torch.gather(ray_z, 1, nearest_ids.unsqueeze(1)).squeeze(1)
    bad_height = base_z - ground_z < min_height

    return torch.where(has_valid_hit, bad_height, torch.zeros_like(bad_height, dtype=torch.bool))
