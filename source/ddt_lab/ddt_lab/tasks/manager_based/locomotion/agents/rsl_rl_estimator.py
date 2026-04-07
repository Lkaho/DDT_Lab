from __future__ import annotations

import copy
import json
import os
import warnings

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import EmpiricalNormalization, MLP


def _diag_limit() -> int:
    return int(os.getenv("DDT_RSL_DIAG_MAX_LOGS_PER_KEY", "8"))


def _diag_threshold(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _obs_term_threshold(set_name: str, group_name: str) -> float:
    default_threshold = _diag_threshold("DDT_RSL_DIAG_LARGE_TERM_THRESHOLD", 50.0)

    per_group_thresholds = {
        "actions": _diag_threshold("DDT_RSL_DIAG_TERM_ACTIONS_THRESHOLD", 25.0),
        "last_action": _diag_threshold("DDT_RSL_DIAG_TERM_LAST_ACTION_THRESHOLD", 25.0),
        "joint_vel": _diag_threshold("DDT_RSL_DIAG_TERM_JOINT_VEL_THRESHOLD", 25.0),
        "base_lin_vel": _diag_threshold("DDT_RSL_DIAG_TERM_BASE_LIN_VEL_THRESHOLD", 20.0),
        "base_lin_vel_xy": _diag_threshold("DDT_RSL_DIAG_TERM_BASE_LIN_VEL_XY_THRESHOLD", 20.0),
        "base_ang_vel": _diag_threshold("DDT_RSL_DIAG_TERM_BASE_ANG_VEL_THRESHOLD", 20.0),
        "velocity_commands": _diag_threshold("DDT_RSL_DIAG_TERM_COMMAND_THRESHOLD", 20.0),
        "joint_pos": _diag_threshold("DDT_RSL_DIAG_TERM_JOINT_POS_THRESHOLD", 20.0),
        "feet_avg_contact_force": _diag_threshold("DDT_RSL_DIAG_TERM_FEET_FORCE_THRESHOLD", 25.0),
        "height_scan": _diag_threshold("DDT_RSL_DIAG_TERM_HEIGHT_SCAN_THRESHOLD", 5.0),
    }
    return per_group_thresholds.get(group_name, default_threshold)


def _rate_limited_diag(owner, key: str, message: str):
    diag_state = getattr(owner, "_ddt_diag_state", None)
    if diag_state is None:
        diag_state = {}
        setattr(owner, "_ddt_diag_state", diag_state)

    count = diag_state.get(key, 0)
    if count >= _diag_limit():
        return

    print(message)
    diag_state[key] = count + 1


def _tensor_stats(tensor: torch.Tensor) -> tuple[int, float, float, float]:
    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    invalid_count = int((~finite_mask).sum().item())
    finite_values = detached[finite_mask]
    if finite_values.numel() == 0:
        return invalid_count, float("nan"), float("nan"), float("nan")
    max_abs = float(finite_values.abs().max().item())
    min_value = float(finite_values.min().item())
    max_value = float(finite_values.max().item())
    return invalid_count, max_abs, min_value, max_value


def _sanitize_tensor(owner, name: str, tensor: torch.Tensor) -> torch.Tensor:
    invalid_count, max_abs, _, _ = _tensor_stats(tensor)
    if invalid_count == 0:
        return tensor

    _rate_limited_diag(
        owner,
        f"sanitize:{name}",
        (
            "[diag][obs] "
            f"sanitized non-finite values in {name}: invalid={invalid_count} "
            f"shape={tuple(tensor.shape)} max_abs_finite={max_abs:.3e}"
        ),
    )
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _log_large_tensor(owner, name: str, tensor: torch.Tensor, threshold: float):
    invalid_count, max_abs, min_value, max_value = _tensor_stats(tensor)
    if invalid_count > 0:
        _rate_limited_diag(
            owner,
            f"invalid:{name}",
            (
                "[diag][tensor] "
                f"non-finite values in {name}: invalid={invalid_count} "
                f"shape={tuple(tensor.shape)}"
            ),
        )
        return

    if max_abs != max_abs or max_abs <= threshold:
        return

    _rate_limited_diag(
        owner,
        f"large:{name}",
        (
            "[diag][tensor] "
            f"large magnitude in {name}: max_abs={max_abs:.3e} "
            f"min={min_value:.3e} max={max_value:.3e} shape={tuple(tensor.shape)}"
        ),
    )


def _log_obs_term(owner, set_name: str, group_name: str, tensor: torch.Tensor):
    _log_large_tensor(owner, f"{set_name}:{group_name}", tensor, _obs_term_threshold(set_name, group_name))


def _log_action_std_anomaly(owner, std: torch.Tensor):
    detached = std.detach()
    non_finite = int((~torch.isfinite(detached)).sum().item())
    non_positive = int((detached <= 0).sum().item())
    if non_finite == 0 and non_positive == 0:
        return

    _, max_abs, min_value, max_value = _tensor_stats(detached)
    _rate_limited_diag(
        owner,
        "action_std",
        (
            "[diag][policy] "
            f"invalid action std detected: non_finite={non_finite} non_positive={non_positive} "
            f"min={min_value:.3e} max={max_value:.3e} max_abs={max_abs:.3e}"
        ),
    )


def _max_obs_group_abs(obs_timestep, env_idx: int, group_names: list[str]) -> tuple[str, float]:
    best_name = "n/a"
    best_abs = float("nan")
    best_invalid = -1
    for group_name in group_names:
        if group_name not in obs_timestep.keys():
            continue
        group_tensor = obs_timestep[group_name][env_idx]
        invalid_count, max_abs, _, _ = _tensor_stats(group_tensor)
        if invalid_count > 0:
            return f"{group_name}(invalid={invalid_count})", max_abs
        if best_invalid < 0 or max_abs > best_abs:
            best_name = group_name
            best_abs = max_abs
            best_invalid = 0
    return best_name, best_abs


class DiagnosticActorCritic(ActorCritic):
    """Default actor-critic with light-weight observation sanitization and diagnostics."""

    def _concat_obs(self, obs, set_name: str) -> torch.Tensor:
        tensors = []
        for group_name in self.obs_groups[set_name]:
            tensor = _sanitize_tensor(self, f"{set_name}:{group_name}", obs[group_name])
            _log_obs_term(self, set_name, group_name, tensor)
            tensors.append(tensor)
        merged = torch.cat(tensors, dim=-1)
        _log_large_tensor(self, f"{set_name}_obs", merged, _diag_threshold("DDT_RSL_DIAG_LARGE_OBS_THRESHOLD", 100.0))
        return merged

    def get_actor_obs(self, obs):
        return self._concat_obs(obs, "policy")

    def get_critic_obs(self, obs):
        return self._concat_obs(obs, "critic")

    def update_distribution(self, obs):
        mean = self.actor(obs)
        _log_large_tensor(self, "action_mean", mean, _diag_threshold("DDT_RSL_DIAG_LARGE_MEAN_THRESHOLD", 100.0))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        _log_action_std_anomaly(self, std)
        self.distribution = Normal(mean, std)


class PPOWithDiagnostics(PPO):
    """Default PPO with minimal return/value diagnostics."""

    def process_env_step(self, obs, rewards, dones, extras):
        if not hasattr(self, "_ddt_raw_rewards") or self._ddt_raw_rewards.shape != self.storage.rewards.shape:
            self._ddt_raw_rewards = torch.zeros_like(self.storage.rewards)
            self._ddt_timeout_flags = torch.zeros_like(self.storage.rewards)
            self._ddt_timeout_bonus = torch.zeros_like(self.storage.rewards)

        if self.storage.step < self.storage.num_transitions_per_env:
            step = self.storage.step
            raw_rewards = rewards.view(-1, 1).detach().to(self.device)
            time_outs = extras.get("time_outs", torch.zeros_like(dones)).view(-1, 1).detach().to(self.device)
            timeout_bonus = torch.zeros_like(raw_rewards)
            if self.transition.values is not None:
                timeout_bonus = self.gamma * self.transition.values.detach() * time_outs

            self._ddt_raw_rewards[step].copy_(raw_rewards)
            self._ddt_timeout_flags[step].copy_(time_outs)
            self._ddt_timeout_bonus[step].copy_(timeout_bonus)

        super().process_env_step(obs, rewards, dones, extras)

    def _log_return_outliers(self, last_values: torch.Tensor, update_index: int):
        returns = self.storage.returns.detach().squeeze(-1)
        invalid_count, max_abs, min_value, max_value = _tensor_stats(returns)
        detail_threshold = _diag_threshold("DDT_RSL_DIAG_RETURN_DETAIL_THRESHOLD", 1000.0)
        if invalid_count == 0 and (max_abs != max_abs or max_abs <= detail_threshold):
            return

        abs_returns = returns.abs()
        if invalid_count > 0:
            abs_returns = torch.where(torch.isfinite(returns), abs_returns, torch.full_like(abs_returns, float("inf")))
        large_count = int((abs_returns > detail_threshold).sum().item())
        common_step_counter = getattr(getattr(self, "env", None), "common_step_counter", None)
        _rate_limited_diag(
            self,
            f"return_summary:{update_index}",
            (
                "[diag][returns] "
                f"update={update_index} common_step={common_step_counter} invalid={invalid_count} "
                f"large_count={large_count} max_abs={max_abs:.3e} min={min_value:.3e} max={max_value:.3e}"
            ),
        )

        top_k = min(5, abs_returns.numel())
        top_values, top_indices = torch.topk(abs_returns.reshape(-1), k=top_k)
        num_envs = self.storage.num_envs
        last_step = self.storage.num_transitions_per_env - 1
        policy_groups = getattr(self.policy, "obs_groups", {}).get("policy", [])
        critic_groups = getattr(self.policy, "obs_groups", {}).get("critic", [])

        for rank, (top_abs, flat_idx) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
            if top_abs <= detail_threshold and invalid_count == 0:
                break

            step = int(flat_idx // num_envs)
            env_idx = int(flat_idx % num_envs)
            next_value = last_values[env_idx] if step == last_step else self.storage.values[step + 1, env_idx]
            obs_timestep = self.storage.observations[step]
            policy_group_name, policy_group_abs = _max_obs_group_abs(obs_timestep, env_idx, policy_groups)
            critic_group_name, critic_group_abs = _max_obs_group_abs(obs_timestep, env_idx, critic_groups)

            raw_reward = float(self._ddt_raw_rewards[step, env_idx, 0].item()) if hasattr(self, "_ddt_raw_rewards") else float("nan")
            timeout_flag = bool(self._ddt_timeout_flags[step, env_idx, 0].item()) if hasattr(self, "_ddt_timeout_flags") else False
            timeout_bonus = (
                float(self._ddt_timeout_bonus[step, env_idx, 0].item()) if hasattr(self, "_ddt_timeout_bonus") else float("nan")
            )
            done_flag = bool(self.storage.dones[step, env_idx, 0].item())
            stored_reward = float(self.storage.rewards[step, env_idx, 0].item())
            return_value = float(self.storage.returns[step, env_idx, 0].item())
            value = float(self.storage.values[step, env_idx, 0].item())
            next_value_scalar = float(next_value[0].item())
            advantage = float(self.storage.advantages[step, env_idx, 0].item())
            action_norm = float(self.storage.actions[step, env_idx].norm().item())
            mu_norm = float(self.storage.mu[step, env_idx].norm().item()) if hasattr(self.storage, "mu") else float("nan")
            sigma_mean = float(self.storage.sigma[step, env_idx].mean().item()) if hasattr(self.storage, "sigma") else float("nan")

            _rate_limited_diag(
                self,
                f"return_detail:{update_index}:{rank}",
                (
                    "[diag][returns] "
                    f"top{rank} step={step} env={env_idx} return={return_value:.3e} advantage={advantage:.3e} "
                    f"value={value:.3e} next_value={next_value_scalar:.3e} raw_reward={raw_reward:.3e} "
                    f"stored_reward={stored_reward:.3e} timeout={int(timeout_flag)} timeout_bonus={timeout_bonus:.3e} "
                    f"done={int(done_flag)} action_norm={action_norm:.3e} mu_norm={mu_norm:.3e} "
                    f"sigma_mean={sigma_mean:.3e} policy_max={policy_group_name}:{policy_group_abs:.3e} "
                    f"critic_max={critic_group_name}:{critic_group_abs:.3e}"
                ),
            )

    def compute_returns(self, obs):
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

        update_index = getattr(self, "_ddt_update_index", 0) + 1
        self._ddt_update_index = update_index
        setattr(self.policy, "_ddt_update_index", update_index)

        _log_large_tensor(
            self,
            "rollout_values",
            self.storage.values,
            _diag_threshold("DDT_RSL_DIAG_LARGE_VALUE_THRESHOLD", 100.0),
        )
        _log_large_tensor(
            self,
            "rollout_returns",
            self.storage.returns,
            _diag_threshold("DDT_RSL_DIAG_LARGE_RETURN_THRESHOLD", 100.0),
        )
        _log_large_tensor(
            self,
            "rollout_advantages",
            self.storage.advantages,
            _diag_threshold("DDT_RSL_DIAG_LARGE_ADV_THRESHOLD", 100.0),
        )
        self._log_return_outliers(last_values, update_index)
        if hasattr(self.storage, "sigma"):
            _log_action_std_anomaly(self, self.storage.sigma)


class ActorCriticWithEstimator(nn.Module):
    """Actor-critic with a jointly trained history MLP for base velocity estimation."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        estimator_hidden_dims=[256, 128],
        estimator_output_dim: int | None = None,
        num_history: int = 3,
        estimated_history_length: int | None = None,
        history_term_dims: list[int] | None = None,
        estimator_lr: float | None = None,
        deploy_share_policy_and_history: bool | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticWithEstimator.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.num_history = num_history
        self.estimated_history_length = estimated_history_length if estimated_history_length is not None else 1
        self.history_term_dims = history_term_dims
        self.estimator_lr = estimator_lr

        self._history_groups = obs_groups.get("history", obs_groups["policy"])
        self._privileged_groups = obs_groups.get("privileged", [])

        num_actor_obs = sum(obs[group_name].shape[-1] for group_name in obs_groups["policy"])
        num_history_obs = sum(obs[group_name].shape[-1] for group_name in self._history_groups)
        num_critic_obs = sum(obs[group_name].shape[-1] for group_name in obs_groups["critic"])
        if estimator_output_dim is None:
            estimator_output_dim = sum(obs[group_name].shape[-1] for group_name in self._privileged_groups)
        if estimator_output_dim <= 0:
            raise ValueError("ActorCriticWithEstimator requires a non-empty privileged observation target.")
        if self.estimated_history_length < 1:
            raise ValueError("estimated_history_length must be greater than or equal to 1.")
        if self.estimated_history_length > 1:
            if self.history_term_dims is None:
                raise ValueError(
                    "ActorCriticWithEstimator requires history_term_dims when estimated_history_length > 1."
                )
            if sum(self.history_term_dims) * self.num_history != num_history_obs:
                raise ValueError(
                    "history_term_dims do not match the flattened history observation size. "
                    f"Expected {num_history_obs}, got {sum(self.history_term_dims) * self.num_history}."
                )

        self.estimator = MLP(num_history_obs, estimator_output_dim, estimator_hidden_dims, activation)
        actor_estimator_obs_dim = estimator_output_dim * self.estimated_history_length
        self.actor = MLP(num_actor_obs + actor_estimator_obs_dim, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        self.num_actor_obs = num_actor_obs
        self.num_history_obs = num_history_obs
        self.num_critic_obs = num_critic_obs
        self.estimator_output_dim = estimator_output_dim
        self.actor_estimator_obs_dim = actor_estimator_obs_dim
        self.num_actions = num_actions
        auto_share_policy_and_history = self.obs_groups["policy"] == self._history_groups and num_actor_obs == num_history_obs
        self.share_policy_and_history = (
            deploy_share_policy_and_history
            if deploy_share_policy_and_history is not None
            else auto_share_policy_and_history
        )

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs + actor_estimator_obs_dim)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

        self.distribution = None
        self._last_estimated_velocity = None
        Normal.set_default_validate_args(False)

        print(f"Estimator MLP: {self.estimator}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    @property
    def estimated_velocity(self):
        return self._last_estimated_velocity

    def _concat_obs(self, obs, group_names: list[str], set_name: str = "obs") -> torch.Tensor:
        tensors = []
        for group_name in group_names:
            tensor = _sanitize_tensor(self, f"{set_name}:{group_name}", obs[group_name])
            _log_obs_term(self, set_name, group_name, tensor)
            tensors.append(tensor)
        merged = torch.cat(tensors, dim=-1)
        _log_large_tensor(self, f"{set_name}_obs", merged, _diag_threshold("DDT_RSL_DIAG_LARGE_OBS_THRESHOLD", 100.0))
        return merged

    def estimate_from_history(self, obs) -> torch.Tensor:
        history_obs = self._concat_obs(obs, self._history_groups, set_name="history")
        self._last_estimated_velocity = self.estimator(history_obs)
        return self._last_estimated_velocity

    def _history_term_major_to_frame_major(self, history_obs: torch.Tensor) -> torch.Tensor:
        """Convert term-major flattened history into frame-major history."""
        if self.history_term_dims is None:
            raise ValueError("history_term_dims must be provided to convert history observations.")

        history_chunks = []
        cursor = 0
        for term_dim in self.history_term_dims:
            block_size = term_dim * self.num_history
            term_history = history_obs[:, cursor : cursor + block_size]
            term_history = term_history.reshape(history_obs.shape[0], self.num_history, term_dim)
            history_chunks.append(term_history)
            cursor += block_size

        if cursor != history_obs.shape[-1]:
            raise ValueError(
                "history_term_dims do not cover the full flattened history observation. "
                f"Consumed {cursor}, total {history_obs.shape[-1]}."
            )

        return torch.cat(history_chunks, dim=-1)

    def _frame_major_to_term_major_history(self, history_frames: torch.Tensor) -> torch.Tensor:
        """Convert frame-major history back to the term-major flattened layout used by the estimator."""
        if self.history_term_dims is None:
            raise ValueError("history_term_dims must be provided to convert history observations.")

        term_histories = []
        cursor = 0
        for term_dim in self.history_term_dims:
            term_history = history_frames[:, :, cursor : cursor + term_dim]
            term_histories.append(term_history.reshape(history_frames.shape[0], self.num_history * term_dim))
            cursor += term_dim

        return torch.cat(term_histories, dim=-1)

    def _estimate_history_features(self, history_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return actor-side estimator features and the latest 2D velocity estimate.

        When estimated_history_length > 1, the actor receives a flattened history of estimator outputs.
        Each historical estimate is generated causally from the available observation history by using a
        num_history-sized sliding window that is left-padded with the oldest available frame.
        """
        if self.estimated_history_length == 1:
            current_estimate = self.estimator(history_obs)
            return current_estimate, current_estimate

        history_frames = self._history_term_major_to_frame_major(history_obs)
        frame_count = history_frames.shape[1]
        output_length = min(self.estimated_history_length, frame_count)
        end_indices = range(frame_count - output_length, frame_count)

        estimator_windows = []
        for end_idx in end_indices:
            start_idx = max(0, end_idx - self.num_history + 1)
            window = history_frames[:, start_idx : end_idx + 1, :]
            pad_len = self.num_history - window.shape[1]
            if pad_len > 0:
                pad = history_frames[:, :1, :].expand(-1, pad_len, -1)
                window = torch.cat([pad, window], dim=1)
            estimator_windows.append(self._frame_major_to_term_major_history(window))

        stacked_windows = torch.stack(estimator_windows, dim=1)
        batch_size = stacked_windows.shape[0]
        estimator_history = self.estimator(stacked_windows.reshape(batch_size * output_length, -1))
        estimator_history = estimator_history.reshape(batch_size, output_length, -1)

        if output_length < self.estimated_history_length:
            pad_count = self.estimated_history_length - output_length
            left_pad = estimator_history[:, :1, :].expand(-1, pad_count, -1)
            estimator_history = torch.cat([left_pad, estimator_history], dim=1)

        current_estimate = estimator_history[:, -1, :]
        actor_features = estimator_history.reshape(batch_size, -1)
        return actor_features, current_estimate

    def get_estimator_targets(self, obs) -> torch.Tensor:
        return self._concat_obs(obs, self._privileged_groups, set_name="privileged")

    def get_actor_obs(self, obs):
        policy_obs = self._concat_obs(obs, self.obs_groups["policy"], set_name="policy")
        history_obs = self._concat_obs(obs, self._history_groups, set_name="history")
        estimated_features, self._last_estimated_velocity = self._estimate_history_features(history_obs)
        return torch.cat([policy_obs, estimated_features], dim=-1)

    def get_critic_obs(self, obs):
        return self._concat_obs(obs, self.obs_groups["critic"], set_name="critic")

    def update_distribution(self, obs):
        mean = self.actor(obs)
        _log_large_tensor(self, "action_mean", mean, _diag_threshold("DDT_RSL_DIAG_LARGE_MEAN_THRESHOLD", 100.0))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        _log_action_std_anomaly(self, std)
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        return self.actor(obs)

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict, strict=True):
        current_state = self.state_dict()
        filtered_state = {}
        shape_mismatches = []

        for key, value in state_dict.items():
            if key in current_state and current_state[key].shape == value.shape:
                filtered_state[key] = value
            elif key in current_state:
                shape_mismatches.append(key)

        missing_keys, unexpected_keys = super().load_state_dict(filtered_state, strict=False)
        fully_loaded = not missing_keys and not unexpected_keys and not shape_mismatches and len(filtered_state) == len(
            current_state
        )

        if not fully_loaded:
            warnings.warn(
                "ActorCriticWithEstimator loaded a partial checkpoint. "
                f"Missing keys: {missing_keys}, unexpected keys: {unexpected_keys}, "
                f"shape mismatches: {shape_mismatches}",
                stacklevel=2,
            )

        return fully_loaded


class EstimatorActorDeployWrapper(nn.Module):
    """Single-engine deploy wrapper that runs estimator and actor internally."""

    def __init__(self, policy: ActorCriticWithEstimator):
        super().__init__()
        self.estimator = copy.deepcopy(policy.estimator)
        self.actor = copy.deepcopy(policy.actor)
        self.actor_obs_normalizer = copy.deepcopy(policy.actor_obs_normalizer)

        self.num_history = int(policy.num_history)
        self.estimated_history_length = int(policy.estimated_history_length)
        self.policy_obs_dim = int(policy.num_actor_obs)
        self.history_obs_dim = int(policy.num_history_obs)
        self.estimator_output_dim = int(policy.estimator_output_dim)
        self.num_actions = int(policy.num_actions)
        self.share_policy_and_history = bool(policy.share_policy_and_history)
        self.history_term_dims = list(policy.history_term_dims) if policy.history_term_dims is not None else None

        if self.estimated_history_length > 1 and self.history_term_dims is None:
            raise ValueError(
                "EstimatorActorDeployWrapper requires history_term_dims when estimated_history_length > 1."
            )

    @property
    def input_dim(self) -> int:
        if self.share_policy_and_history:
            return self.history_obs_dim
        return self.policy_obs_dim + self.history_obs_dim

    def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.share_policy_and_history:
            return obs, obs
        policy_obs = obs[:, : self.policy_obs_dim]
        history_obs = obs[:, self.policy_obs_dim :]
        return policy_obs, history_obs

    def _history_term_major_to_frame_major(self, history_obs: torch.Tensor) -> torch.Tensor:
        if self.history_term_dims is None:
            raise ValueError("history_term_dims must be provided to convert history observations.")

        history_chunks = []
        cursor = 0
        for term_dim in self.history_term_dims:
            block_size = term_dim * self.num_history
            term_history = history_obs[:, cursor : cursor + block_size]
            term_history = term_history.reshape(history_obs.shape[0], self.num_history, term_dim)
            history_chunks.append(term_history)
            cursor += block_size

        if cursor != history_obs.shape[-1]:
            raise ValueError(
                "history_term_dims do not cover the full flattened history observation. "
                f"Consumed {cursor}, total {history_obs.shape[-1]}."
            )

        return torch.cat(history_chunks, dim=-1)

    def _frame_major_to_term_major_history(self, history_frames: torch.Tensor) -> torch.Tensor:
        if self.history_term_dims is None:
            raise ValueError("history_term_dims must be provided to convert history observations.")

        term_histories = []
        cursor = 0
        for term_dim in self.history_term_dims:
            term_history = history_frames[:, :, cursor : cursor + term_dim]
            term_histories.append(term_history.reshape(history_frames.shape[0], self.num_history * term_dim))
            cursor += term_dim
        return torch.cat(term_histories, dim=-1)

    def _estimate_history_features(self, history_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.estimated_history_length == 1:
            current_estimate = self.estimator(history_obs)
            return current_estimate, current_estimate

        history_frames = self._history_term_major_to_frame_major(history_obs)
        frame_count = history_frames.shape[1]
        output_length = min(self.estimated_history_length, frame_count)
        end_indices = range(frame_count - output_length, frame_count)

        estimator_windows = []
        for end_idx in end_indices:
            start_idx = max(0, end_idx - self.num_history + 1)
            window = history_frames[:, start_idx : end_idx + 1, :]
            pad_len = self.num_history - window.shape[1]
            if pad_len > 0:
                pad = history_frames[:, :1, :].expand(-1, pad_len, -1)
                window = torch.cat([pad, window], dim=1)
            estimator_windows.append(self._frame_major_to_term_major_history(window))

        stacked_windows = torch.stack(estimator_windows, dim=1)
        batch_size = stacked_windows.shape[0]
        estimator_history = self.estimator(stacked_windows.reshape(batch_size * output_length, -1))
        estimator_history = estimator_history.reshape(batch_size, output_length, -1)

        if output_length < self.estimated_history_length:
            pad_count = self.estimated_history_length - output_length
            left_pad = estimator_history[:, :1, :].expand(-1, pad_count, -1)
            estimator_history = torch.cat([left_pad, estimator_history], dim=1)

        current_estimate = estimator_history[:, -1, :]
        return estimator_history.reshape(batch_size, -1), current_estimate

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy_obs, history_obs = self._split_obs(obs)
        estimated_features, current_estimate = self._estimate_history_features(history_obs)
        actor_obs = torch.cat([policy_obs, estimated_features], dim=-1)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        return self.actor(actor_obs), current_estimate


def is_estimator_policy(policy: nn.Module) -> bool:
    return isinstance(policy, ActorCriticWithEstimator)


def get_estimator_deploy_metadata(policy: ActorCriticWithEstimator) -> dict[str, object]:
    deploy_wrapper = EstimatorActorDeployWrapper(policy)
    return {
        "type": "estimator_actor_deploy_wrapper",
        "input_name": "obs",
        "output_names": ["actions", "estimated_velocity"],
        "input_dim": deploy_wrapper.input_dim,
        "output_dim": deploy_wrapper.num_actions,
        "estimated_velocity_dim": deploy_wrapper.estimator_output_dim,
        "policy_obs_dim": deploy_wrapper.policy_obs_dim,
        "history_obs_dim": deploy_wrapper.history_obs_dim,
        "estimator_output_dim": deploy_wrapper.estimator_output_dim,
        "estimated_history_length": deploy_wrapper.estimated_history_length,
        "actor_estimator_obs_dim": deploy_wrapper.estimator_output_dim * deploy_wrapper.estimated_history_length,
        "share_policy_and_history": deploy_wrapper.share_policy_and_history,
        "input_layout": "history_only" if deploy_wrapper.share_policy_and_history else "policy_then_history",
        "num_history": deploy_wrapper.num_history,
        "history_term_dims": deploy_wrapper.history_term_dims,
    }


def export_estimator_policy_metadata(
    policy: ActorCriticWithEstimator, path: str, filename: str = "policy_metadata.json"
) -> None:
    os.makedirs(path, exist_ok=True)
    export_path = os.path.join(path, filename)
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(get_estimator_deploy_metadata(policy), f, indent=2)


def export_estimator_policy_as_jit(policy: ActorCriticWithEstimator, path: str, filename: str = "policy.pt") -> None:
    os.makedirs(path, exist_ok=True)
    deploy_wrapper = EstimatorActorDeployWrapper(policy)
    deploy_wrapper.to("cpu")
    deploy_wrapper.eval()
    example_obs = torch.zeros(1, deploy_wrapper.input_dim)
    traced_module = torch.jit.trace(deploy_wrapper, example_obs)
    traced_module.save(os.path.join(path, filename))


def export_estimator_policy_as_onnx(
    policy: ActorCriticWithEstimator, path: str, filename: str = "policy.onnx", verbose: bool = False
) -> None:
    os.makedirs(path, exist_ok=True)
    deploy_wrapper = EstimatorActorDeployWrapper(policy)
    deploy_wrapper.to("cpu")
    deploy_wrapper.eval()
    example_obs = torch.zeros(1, deploy_wrapper.input_dim)
    torch.onnx.export(
        deploy_wrapper,
        example_obs,
        os.path.join(path, filename),
        export_params=True,
        opset_version=18,
        verbose=verbose,
        input_names=["obs"],
        output_names=["actions", "estimated_velocity"],
        dynamic_axes={},
    )


class PPOWithEstimator(PPO):
    """PPO with an auxiliary supervised loss for the history-based estimator."""

    def __init__(self, *args, estimator_loss_coef: float = 1.0, estimator_lr: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimator_loss_coef = estimator_loss_coef
        self.estimator_lr = estimator_lr

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_estimator_loss = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        mse_loss = torch.nn.MSELoss()

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            estimator_loss = torch.zeros((), device=self.device)
            if self.policy.obs_groups.get("privileged"):
                predicted_privileged = self.policy.estimate_from_history(obs_batch)
                target_privileged = self.policy.get_estimator_targets(obs_batch).detach()
                estimator_loss = mse_loss(predicted_privileged, target_privileged)
                loss = loss + self.estimator_loss_coef * estimator_loss

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss = loss + self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                rnd_loss = mse_loss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_estimator_loss += estimator_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_estimator_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "estimator": mean_estimator_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        return loss_dict


def register_rsl_rl_estimator_extensions():
    """Register estimator-aware policy/algo classes into rsl_rl runner namespaces."""
    import rsl_rl.algorithms as rsl_algorithms
    import rsl_rl.modules as rsl_modules
    import rsl_rl.runners.on_policy_runner as on_policy_runner_module

    rsl_modules.ActorCritic = DiagnosticActorCritic
    rsl_modules.ActorCriticWithEstimator = ActorCriticWithEstimator
    rsl_algorithms.PPO = PPOWithDiagnostics
    rsl_algorithms.PPOWithEstimator = PPOWithEstimator
    on_policy_runner_module.ActorCritic = DiagnosticActorCritic
    on_policy_runner_module.ActorCriticWithEstimator = ActorCriticWithEstimator
    on_policy_runner_module.PPO = PPOWithDiagnostics
    on_policy_runner_module.PPOWithEstimator = PPOWithEstimator
