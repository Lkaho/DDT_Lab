from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.envs.mdp.actions import actions_cfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class JointPositionWithFeedforwardAction(JointPositionAction):
    """Joint position action with contact-triggered feedforward trajectory injection."""

    cfg: "JointPositionWithFeedforwardActionCfg"

    def __init__(self, cfg: "JointPositionWithFeedforwardActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._ff_enabled = cfg.feedforward_enabled
        self._k_fb = cfg.k_fb
        self._k_ff = cfg.k_ff
        self._initial_k_ff = cfg.k_ff
        self._ff_period = cfg.feedforward_period
        self._contact_trigger_enabled = cfg.contact_trigger_enabled
        self._force_threshold = cfg.contact_force_threshold
        self._followup_trigger_delay = cfg.followup_trigger_delay_factor * self._ff_period
        self._k_ff_anneal_enabled = cfg.k_ff_anneal_enabled
        self._k_ff_final = cfg.k_ff_final
        self._k_ff_start_iteration = cfg.k_ff_start_iteration
        self._k_ff_anneal_iterations = cfg.k_ff_anneal_iterations
        self._k_ff_steps_per_iteration = max(1, cfg.k_ff_steps_per_iteration)

        self._ff_amplitude = torch.zeros(self._num_joints, device=self.device)
        if isinstance(cfg.feedforward_amplitude, dict):
            import re

            for i, name in enumerate(self._joint_names):
                for pattern, amplitude in cfg.feedforward_amplitude.items():
                    if re.match(pattern, name):
                        self._ff_amplitude[i] = amplitude
                        break
        else:
            self._ff_amplitude[:] = float(cfg.feedforward_amplitude)

        if cfg.feedforward_joint_names is not None:
            ff_joint_ids, ff_joint_names = self._asset.find_joints(cfg.feedforward_joint_names)
            if isinstance(self._joint_ids, slice):
                joint_ids_list = list(range(self._asset.num_joints))
            else:
                joint_ids_list = list(self._joint_ids)
            self._ff_local_ids = torch.tensor(
                [joint_ids_list.index(joint_id) for joint_id in ff_joint_ids if joint_id in joint_ids_list],
                device=self.device,
                dtype=torch.long,
            )
            self._ff_leg_mapping = torch.tensor(
                [0 if "right" in name.lower() else 1 for name in ff_joint_names],
                device=self.device,
                dtype=torch.long,
            )
        else:
            self._ff_local_ids = torch.arange(self._num_joints, device=self.device, dtype=torch.long)
            self._ff_leg_mapping = torch.zeros(self._num_joints, device=self.device, dtype=torch.long)

        self._time = torch.zeros(self.num_envs, 2, device=self.device)
        self._contact_sensor = None
        self._lifting_state = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._first_leg = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_lift_signal = torch.zeros(self.num_envs, 2, device=self.device)
        self._last_ff_signal = torch.zeros(self.num_envs, 2, device=self.device)
        self._last_trigger_signal = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._last_ff_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._last_ff_contribution = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._last_blended_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    @property
    def lifting_state(self) -> torch.Tensor:
        return self._lifting_state

    @property
    def k_ff(self) -> float:
        return float(self._k_ff)

    @property
    def initial_k_ff(self) -> float:
        return float(self._initial_k_ff)

    @property
    def lift_signal(self) -> torch.Tensor:
        return self._last_lift_signal

    @property
    def ff_signal(self) -> torch.Tensor:
        return self._last_ff_signal

    @property
    def trigger_signal(self) -> torch.Tensor:
        return self._last_trigger_signal

    @property
    def ff_actions(self) -> torch.Tensor:
        return self._last_ff_actions

    @property
    def ff_contribution(self) -> torch.Tensor:
        return self._last_ff_contribution

    @property
    def blended_actions(self) -> torch.Tensor:
        return self._last_blended_actions

    @property
    def controlled_joint_ids(self) -> torch.Tensor:
        if isinstance(self._joint_ids, slice):
            return torch.arange(self._asset.num_joints, device=self.device, dtype=torch.long)
        return torch.as_tensor(self._joint_ids, device=self.device, dtype=torch.long)

    @property
    def ff_joint_local_ids(self) -> torch.Tensor:
        return self._ff_local_ids

    @property
    def ff_target_positions(self) -> torch.Tensor:
        return self._last_ff_actions + self._offset

    def _update_k_ff_schedule(self):
        if not (self._ff_enabled and self._k_ff_anneal_enabled):
            return

        current_iteration = self._env.common_step_counter // self._k_ff_steps_per_iteration
        if current_iteration < self._k_ff_start_iteration:
            current_k_ff = self._initial_k_ff
        else:
            progress = min(
                1.0,
                (current_iteration - self._k_ff_start_iteration) / max(1, self._k_ff_anneal_iterations),
            )
            current_k_ff = self._initial_k_ff + (self._k_ff_final - self._initial_k_ff) * progress
        self._k_ff = float(current_k_ff)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._update_k_ff_schedule()
        self._last_lift_signal.zero_()
        self._last_ff_signal.zero_()
        self._last_trigger_signal.zero_()
        self._last_ff_actions.zero_()
        self._last_ff_contribution.zero_()

        if self._ff_enabled:
            if self._contact_trigger_enabled:
                lift_signal = self._compute_lift_signal()
            else:
                lift_signal = torch.ones(self.num_envs, 2, device=self.device)
                self._last_trigger_signal.zero_()

            self._last_lift_signal[:] = lift_signal
            self._time += self._env.step_dt * lift_signal
            if self._k_ff > 0.0:
                phase = 2.0 * math.pi * self._time / self._ff_period
                ff_signal = 0.5 * (1.0 - torch.cos(phase))
                self._last_ff_signal[:] = ff_signal

                ff_actions = torch.zeros_like(self._raw_actions)
                for i, local_id in enumerate(self._ff_local_ids):
                    leg_idx = self._ff_leg_mapping[i]
                    ff_actions[:, local_id] = (
                        ff_signal[:, leg_idx]
                        * self._ff_amplitude[local_id]
                        * lift_signal[:, leg_idx]
                    )

                self._last_ff_actions[:] = ff_actions
                self._last_ff_contribution[:] = self._k_ff * ff_actions
                blended_actions = self._k_fb * self._raw_actions + self._k_ff * ff_actions
            else:
                blended_actions = self._raw_actions
        else:
            blended_actions = self._raw_actions

        self._last_blended_actions[:] = blended_actions
        self._processed_actions = blended_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._time[env_ids] = 0.0
        self._lifting_state[env_ids] = False
        self._first_leg[env_ids] = 0
        self._last_lift_signal[env_ids] = 0.0
        self._last_ff_signal[env_ids] = 0.0
        self._last_trigger_signal[env_ids] = False
        self._last_ff_actions[env_ids] = 0.0
        self._last_ff_contribution[env_ids] = 0.0
        self._last_blended_actions[env_ids] = 0.0
        if hasattr(self, "_contact_force_history"):
            self._contact_force_history[:, env_ids] = 0.0

    def _compute_lift_signal(self) -> torch.Tensor:
        if self._contact_sensor is None:
            from isaaclab.sensors import ContactSensor
            import re

            self._contact_sensor: ContactSensor = self._env.scene.sensors[self.cfg.contact_sensor_name]
            self._right_foot_id = None
            self._left_foot_id = None
            for i, name in enumerate(self._contact_sensor.body_names):
                if re.match(self.cfg.contact_body_pattern, name):
                    if "right" in name.lower():
                        self._right_foot_id = i
                    elif "left" in name.lower():
                        self._left_foot_id = i
            self._contact_force_history = torch.zeros(3, self.num_envs, 2, device=self.device)

        forces_xyz = self._contact_sensor.data.net_forces_w
        right_xy = torch.zeros(self.num_envs, device=self.device)
        left_xy = torch.zeros(self.num_envs, device=self.device)
        if self._right_foot_id is not None:
            right_force = forces_xyz[:, self._right_foot_id, :]
            right_xy = torch.sqrt(right_force[:, 0] ** 2 + right_force[:, 1] ** 2)
        if self._left_foot_id is not None:
            left_force = forces_xyz[:, self._left_foot_id, :]
            left_xy = torch.sqrt(left_force[:, 0] ** 2 + left_force[:, 1] ** 2)

        feet_xy = torch.stack([right_xy, left_xy], dim=1)
        self._contact_force_history[:-1] = self._contact_force_history[1:].clone()
        self._contact_force_history[-1] = feet_xy

        avg_force = self._contact_force_history.mean(dim=0)
        right_contact = avg_force[:, 0] > self._force_threshold
        left_contact = avg_force[:, 1] > self._force_threshold

        stable_contact = (self._contact_force_history > self._force_threshold).all(dim=0)
        right_stable = stable_contact[:, 0]
        left_stable = stable_contact[:, 1]

        trigger_right = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        trigger_left = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        no_leg_lifting = ~self._lifting_state[:, 0] & ~self._lifting_state[:, 1]
        left_followup_ready = (self._first_leg == 1) & self._lifting_state[:, 0] & ~self._lifting_state[:, 1]
        right_followup_ready = (self._first_leg == 2) & self._lifting_state[:, 1] & ~self._lifting_state[:, 0]
        can_trigger_right = ~self._lifting_state[:, 0] & (no_leg_lifting | right_followup_ready)
        can_trigger_left = ~self._lifting_state[:, 1] & (no_leg_lifting | left_followup_ready)

        right_only_eligible = can_trigger_right & ~can_trigger_left
        left_only_eligible = can_trigger_left & ~can_trigger_right
        trigger_right = trigger_right | (right_only_eligible & right_contact)
        trigger_left = trigger_left | (left_only_eligible & left_contact)

        both_eligible = can_trigger_right & can_trigger_left
        only_right = right_contact & ~left_contact & both_eligible
        only_left = left_contact & ~right_contact & both_eligible
        trigger_right = trigger_right | only_right
        trigger_left = trigger_left | only_left

        both_contact = right_contact & left_contact & both_eligible
        trigger_right = trigger_right | (both_contact & right_stable & ~left_stable)
        trigger_left = trigger_left | (both_contact & ~right_stable & left_stable)

        both_stable = both_contact & right_stable & left_stable
        trigger_right = trigger_right | (both_stable & (avg_force[:, 0] >= avg_force[:, 1]))
        trigger_left = trigger_left | (both_stable & (avg_force[:, 0] < avg_force[:, 1]))
        self._last_trigger_signal[:, 0] = trigger_right
        self._last_trigger_signal[:, 1] = trigger_left

        self._lifting_state[:, 0] = self._lifting_state[:, 0] | trigger_right
        self._lifting_state[:, 1] = self._lifting_state[:, 1] | trigger_left

        self._first_leg = torch.where(
            trigger_right & (self._first_leg == 0), torch.ones_like(self._first_leg), self._first_leg
        )
        self._first_leg = torch.where(
            trigger_left & (self._first_leg == 0), torch.full_like(self._first_leg, 2), self._first_leg
        )

        right_done = self._time[:, 0] >= self._ff_period
        left_done = self._time[:, 1] >= self._ff_period
        self._time[:, 0] = torch.where(right_done, torch.zeros_like(self._time[:, 0]), self._time[:, 0])
        self._time[:, 1] = torch.where(left_done, torch.zeros_like(self._time[:, 1]), self._time[:, 1])
        self._lifting_state[:, 0] = torch.where(
            right_done, torch.zeros_like(self._lifting_state[:, 0]), self._lifting_state[:, 0]
        )
        self._lifting_state[:, 1] = torch.where(
            left_done, torch.zeros_like(self._lifting_state[:, 1]), self._lifting_state[:, 1]
        )

        no_active_lifts = ~self._lifting_state.any(dim=1)
        self._first_leg = torch.where(no_active_lifts, torch.zeros_like(self._first_leg), self._first_leg)
        return self._lifting_state.float()


@configclass
class JointPositionWithFeedforwardActionCfg(ActionTermCfg):
    """Configuration for joint position action with contact-triggered feedforward support."""

    class_type: type[ActionTerm] = JointPositionWithFeedforwardAction

    joint_names: list[str] = MISSING
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    preserve_order: bool = False
    use_default_offset: bool = True
    clip: dict[str, tuple[float, float]] | None = None

    feedforward_enabled: bool = False
    k_fb: float = 1.0
    k_ff: float = 0.0
    feedforward_period: float = 0.6
    feedforward_amplitude: float | dict[str, float] = 0.0
    feedforward_joint_names: list[str] | None = None
    contact_trigger_enabled: bool = False
    contact_sensor_name: str = "contact_forces"
    contact_body_pattern: str = ".*_leg_4"
    contact_force_threshold: float = 10.0
    followup_trigger_delay_factor: float = 0.0
    k_ff_anneal_enabled: bool = False
    k_ff_final: float = 0.0
    k_ff_start_iteration: int = 0
    k_ff_anneal_iterations: int = 0
    k_ff_steps_per_iteration: int = 24
