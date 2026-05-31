# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cost manager for constrained PPO-style training."""

from __future__ import annotations

import torch
from prettytable import PrettyTable

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg
from isaaclab.utils import configclass


@configclass
class CostTermCfg(ManagerTermBaseCfg):
    """Configuration for one non-negative training cost term."""

    scale: float = 1.0
    """Scale applied to the raw cost value."""

    d_value: float = 0.0
    """Safety budget used by the Lagrangian violation loss."""

    k_value: float = 0.01
    """Initial Lagrangian multiplier."""


class CostManager:
    """Computes per-step cost vectors and episode cost logs."""

    def __init__(self, cfg, env, device: str | torch.device):
        self._cfg = cfg
        self._env = env
        self._device = device

        self._terms: list[tuple[str, CostTermCfg]] = []
        for name in dir(cfg):
            if name.startswith("_"):
                continue
            term = getattr(cfg, name)
            if not isinstance(term, CostTermCfg):
                continue
            for value in term.params.values():
                if isinstance(value, SceneEntityCfg):
                    value.resolve(env.scene)
            self._terms.append((name, term))

        if not self._terms:
            raise ValueError(
                f"CostsCfg '{type(cfg).__name__}' has no CostTermCfg attributes. "
                "Remove the cfg or add at least one cost term."
            )

        self.num_costs = len(self._terms)
        self.active_terms = [name for name, _ in self._terms]
        self.k_values = torch.tensor([term.k_value for _, term in self._terms], device=device).view(1, -1)
        self.d_values_tensor = torch.tensor([term.d_value for _, term in self._terms], device=device).view(1, 1, -1)
        self._episode_sums = {
            name: torch.zeros(env.num_envs, device=device, dtype=torch.float32) for name in self.active_terms
        }

        print("[INFO] Cost Manager: ", self)

    def __str__(self) -> str:
        msg = f"<CostManager> contains {self.num_costs} active terms.\n"
        table = PrettyTable()
        table.title = "Active Cost Terms"
        table.field_names = ["Index", "Name", "Scale", "d_value", "k_value"]
        table.align["Name"] = "l"
        for idx, (name, term) in enumerate(self._terms):
            table.add_row([idx, name, f"{term.scale:.4g}", f"{term.d_value:.4g}", f"{term.k_value:.4g}"])
        return msg + table.get_string() + "\n"

    @torch.no_grad()
    def compute(self) -> torch.Tensor:
        """Return costs with shape ``(num_envs, num_costs)``."""
        costs = []
        for name, term in self._terms:
            value = (term.func(self._env, **term.params) * term.scale).clamp_min_(0.0)
            self._episode_sums[name] += value
            costs.append(value.unsqueeze(-1))
        return torch.cat(costs, dim=-1)

    @torch.no_grad()
    def log_episode(self, env_ids: torch.Tensor, max_episode_length_s: float) -> dict[str, torch.Tensor]:
        """Return and clear per-second episode costs for completed envs."""
        log = {}
        for name in self.active_terms:
            sums = self._episode_sums[name]
            log[f"Episode_Cost/{name}"] = sums[env_ids].mean() / max_episode_length_s
            sums[env_ids] = 0.0
        return log
