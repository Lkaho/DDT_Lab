# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.envs import ManagerBasedRLEnv

from ddt_lab.managers import CostManager


class PositiveRewardManagerBasedRLEnv(ManagerBasedRLEnv):
    """Manager-based RL env with optional reward clipping and constrained-training costs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cost_cfg = getattr(self.cfg, "costs", None)
        self.cost_manager = CostManager(cost_cfg, self, self.device) if cost_cfg is not None else None

    def step(self, action: torch.Tensor):
        obs, rewards, terminated, time_outs, extras = super().step(action)
        if self.cost_manager is not None:
            costs = self.cost_manager.compute()
            extras["costs"] = costs
            extras["cost_names"] = self.cost_manager.active_terms
            extras["cost_k_values"] = self.cost_manager.k_values
            extras["cost_d_values"] = self.cost_manager.d_values_tensor
            done_ids = torch.nonzero(terminated | time_outs, as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                episode_length_s = getattr(self, "max_episode_length_s", self.cfg.episode_length_s)
                extras.setdefault("log", {}).update(self.cost_manager.log_episode(done_ids, episode_length_s))
        if getattr(self.cfg, "only_positive_rewards", False):
            self.reward_buf.clamp_min_(0.0)
            rewards = self.reward_buf
        return obs, rewards, terminated, time_outs, extras
