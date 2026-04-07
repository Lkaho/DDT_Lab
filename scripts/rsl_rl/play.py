# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Use keyboard to drive base_velocity.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import re
import time
import math

import ddt_lab.tasks  # noqa: F401
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.utils import math as math_utils
from ddt_lab.tasks.manager_based.locomotion.agents.rsl_rl_estimator import (
    export_estimator_policy_as_jit,
    export_estimator_policy_as_onnx,
    export_estimator_policy_metadata,
    is_estimator_policy,
    register_rsl_rl_estimator_extensions,
)
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

register_rsl_rl_estimator_extensions()


def _resolve_checkpoint_iteration(runner: OnPolicyRunner | DistillationRunner, resume_path: str) -> int | None:
    """Resolve the training iteration stored in the loaded checkpoint."""
    current_iteration = getattr(runner, "current_learning_iteration", None)
    if isinstance(current_iteration, int):
        return current_iteration

    match = re.search(r"model_(\d+)\.pt$", os.path.basename(resume_path))
    if match is not None:
        return int(match.group(1))

    return None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    keyboard_interface = None
    keyboard_command_state = {
        "command": None,
        "raw_command": None,
        "desired_heading": None,
        "hold_yaw_command": None,
    }
    if args_cli.keyboard:
        if getattr(args_cli, "headless", False):
            raise ValueError("--keyboard requires a live simulator window. Please run play.py without --headless.")

        env_cfg.scene.num_envs = 1
        if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
            env_cfg.commands.base_velocity.debug_vis = False

        keyboard_sensitivity = 0.4
        keyboard_cfg = Se2KeyboardCfg(
            v_x_sensitivity=keyboard_sensitivity,
            v_y_sensitivity=keyboard_sensitivity,
            omega_z_sensitivity=keyboard_sensitivity,
            sim_device=env_cfg.sim.device,
        )
        keyboard_interface = Se2Keyboard(keyboard_cfg)

        def _keyboard_velocity_command(env):
            command = keyboard_command_state["command"]
            if command is None:
                command = keyboard_interface.advance().to(env.device)
                if command.ndim == 1:
                    command = command.unsqueeze(0)
                keyboard_command_state["command"] = command
            return command

        observations_cfg = getattr(env_cfg, "observations", None)
        if observations_cfg is not None:
            for group_name in ("policy", "history"):
                obs_group_cfg = getattr(observations_cfg, group_name, None)
                if obs_group_cfg is not None and hasattr(obs_group_cfg, "enable_corruption"):
                    obs_group_cfg.enable_corruption = False
                if obs_group_cfg is not None and hasattr(obs_group_cfg, "velocity_commands"):
                    velocity_term_cfg = getattr(obs_group_cfg, "velocity_commands")
                    setattr(
                        obs_group_cfg,
                        "velocity_commands",
                        ObsTerm(
                            func=_keyboard_velocity_command,
                            scale=getattr(velocity_term_cfg, "scale", 1.0),
                        ),
                    )

        events_cfg = getattr(env_cfg, "events", None)
        if events_cfg is not None:
            for event_name in (
                "physics_material",
                "add_base_mass",
                "add_base_inertia",
                "add_base_com",
                "base_external_force_torque",
                "randomize_actuator_gains",
                "push_robot",
            ):
                if hasattr(events_cfg, event_name):
                    setattr(events_cfg, event_name, None)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    policy_term_names = env.unwrapped.observation_manager.active_terms.get("policy", [])
    policy_term_cfgs = env.unwrapped.observation_manager._group_obs_term_cfgs.get("policy", [])
    privileged_term_names = env.unwrapped.observation_manager.active_terms.get("privileged", [])
    privileged_term_cfgs = env.unwrapped.observation_manager._group_obs_term_cfgs.get("privileged", [])
    estimator_target_scale = 1.0
    if "base_lin_vel_xy" in privileged_term_names:
        privileged_idx = privileged_term_names.index("base_lin_vel_xy")
        privileged_cfg = privileged_term_cfgs[privileged_idx]
        if isinstance(privileged_cfg.scale, (float, int)):
            estimator_target_scale = float(privileged_cfg.scale)

    for term_name, term_cfg in zip(policy_term_names, policy_term_cfgs):
        if term_name not in {"joint_pos", "joint_vel"}:
            continue
        asset_cfg = term_cfg.params.get("asset_cfg")
        if asset_cfg is None:
            print(f"[INFO] policy/{term_name} resolved asset_cfg: None")
            continue
        asset = env.unwrapped.scene[asset_cfg.name]
        if asset_cfg.joint_ids == slice(None):
            joint_ids = list(range(len(asset.joint_names)))
        else:
            joint_ids = list(asset_cfg.joint_ids)
        joint_names = [asset.joint_names[i] for i in joint_ids]
        print(f"[INFO] policy/{term_name} resolved joint_ids: {joint_ids}")
        print(f"[INFO] policy/{term_name} resolved joint_names: {joint_names}")
        print(f"[INFO] policy/{term_name} preserve_order: {asset_cfg.preserve_order}")

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    joint_pos_term = None
    if "joint_pos" in env.unwrapped.action_manager.active_terms:
        joint_pos_term = env.unwrapped.action_manager.get_term("joint_pos")

    apply_keyboard_command = None
    if args_cli.keyboard:
        if "base_velocity" not in env.unwrapped.command_manager.active_terms:
            raise ValueError("--keyboard requires an active 'base_velocity' command term.")

        base_velocity_term = env.unwrapped.command_manager.get_term("base_velocity")
        print("[INFO] Keyboard teleoperation enabled for base_velocity.")
        print(keyboard_interface)
        print("[INFO] Keyboard mode: disabled observation corruption and domain-randomization events.")
        print("[INFO] Keyboard mode: using deployment-style heading hold when no yaw key is pressed.")

        def _apply_keyboard_command():
            raw_command = keyboard_interface.advance().to(env.unwrapped.device)
            if raw_command.ndim == 1:
                raw_command = raw_command.unsqueeze(0)

            current_heading = base_velocity_term.robot.data.heading_w.clone()
            desired_heading = keyboard_command_state["desired_heading"]
            if desired_heading is None:
                desired_heading = current_heading.clone()

            user_yaw_rate = raw_command[:, 2]
            active_yaw_input = torch.abs(user_yaw_rate) > 1.0e-3
            desired_heading = torch.where(active_yaw_input, current_heading, desired_heading)

            yaw_error = math_utils.wrap_to_pi(desired_heading - current_heading)
            hold_yaw_command = torch.clamp(
                getattr(base_velocity_term.cfg, "heading_control_stiffness", 0.5) * yaw_error,
                min=base_velocity_term.cfg.ranges.ang_vel_z[0],
                max=base_velocity_term.cfg.ranges.ang_vel_z[1],
            )
            final_command = raw_command.clone()
            final_command[:, 2] = torch.where(active_yaw_input, user_yaw_rate, hold_yaw_command)

            keyboard_command_state["raw_command"] = raw_command
            keyboard_command_state["command"] = final_command
            keyboard_command_state["desired_heading"] = desired_heading
            keyboard_command_state["hold_yaw_command"] = hold_yaw_command

            base_velocity_term.vel_command_b[:] = final_command
            if hasattr(base_velocity_term, "is_heading_env"):
                base_velocity_term.is_heading_env[:] = False
            if hasattr(base_velocity_term, "is_standing_env"):
                base_velocity_term.is_standing_env[:] = False
            if hasattr(base_velocity_term, "heading_target"):
                base_velocity_term.heading_target[:] = desired_heading
            if hasattr(base_velocity_term, "_apply_terrain_command_filter"):
                base_velocity_term._apply_terrain_command_filter()

        apply_keyboard_command = _apply_keyboard_command

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    checkpoint_iteration = _resolve_checkpoint_iteration(runner, resume_path)
    if joint_pos_term is not None and checkpoint_iteration is not None:
        steps_per_iteration = max(1, int(getattr(joint_pos_term, "_k_ff_steps_per_iteration", 1)))
        env.unwrapped.common_step_counter = checkpoint_iteration * steps_per_iteration
        if hasattr(joint_pos_term, "_update_k_ff_schedule"):
            joint_pos_term._update_k_ff_schedule()
        if hasattr(joint_pos_term, "k_ff"):
            print(
                "[INFO] Play mode: initialized joint_pos k_ff from checkpoint "
                f"iteration {checkpoint_iteration} -> k_ff={joint_pos_term.k_ff:.4f}."
            )
    elif joint_pos_term is not None:
        print("[INFO] Play mode: could not resolve checkpoint iteration, keeping joint_pos k_ff as configured.")

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if is_estimator_policy(policy_nn):
        export_estimator_policy_as_jit(policy_nn, path=export_model_dir, filename="policy.pt")
        export_estimator_policy_as_onnx(policy_nn, path=export_model_dir, filename="policy.onnx")
        export_estimator_policy_metadata(policy_nn, path=export_model_dir, filename="policy_metadata.json")
        print(f"[INFO] Exported single-engine estimator deploy policy to: {export_model_dir}")
    else:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    print_interval = int(0.5 / dt)  # print every 0.5 seconds
    print(f"[INFO] Print interval: every {print_interval} steps (0.5s)")

    # reset environment
    obs = env.get_observations()
    if apply_keyboard_command is not None:
        apply_keyboard_command()
        obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if apply_keyboard_command is not None:
                apply_keyboard_command()
                obs = env.get_observations()

            # agent stepping
            obs_for_policy = obs
            actions = policy(obs_for_policy)
            
            # env stepping
            obs, _, _, _ = env.step(actions)
            
            # print every 0.5 seconds (after step to get processed actions)
            if timestep % print_interval == 0:
                # Get action info from action_manager
                action_manager = env.unwrapped.action_manager
                action_values = actions[0].cpu().numpy()

                print(f"\n[Step {timestep}, Time {timestep * dt:.1f}s]")
                if args_cli.keyboard:
                    if "policy" in obs_for_policy.keys():
                        actor_obs_tensor = obs_for_policy["policy"]
                        actor_obs = actor_obs_tensor[0].detach().cpu().numpy()
                    else:
                        actor_obs = obs_for_policy[0].detach().cpu().numpy()

                    obs_len = len(actor_obs)
                    policy_term_dims = env.unwrapped.observation_manager.group_obs_term_dim.get("policy", [])
                    print(f"  Current policy observation: total_dim={obs_len}")
                    print("  Policy terms (current frame only):")

                    idx = 0
                    for term_name, term_dim, term_cfg in zip(policy_term_names, policy_term_dims, policy_term_cfgs):
                        flattened_dim = math.prod(term_dim)
                        term_history = term_cfg.history_length if term_cfg.history_length is not None else 1
                        base_dim = flattened_dim // term_history if term_history > 1 else flattened_dim
                        total_dim = flattened_dim
                        if idx + total_dim > obs_len:
                            break
                        term_values = actor_obs[idx : idx + total_dim]
                        current_vals = term_values[-base_dim:] if term_history > 1 else term_values
                        print(f"    {term_name}: {current_vals}")
                        idx += total_dim

                if hasattr(policy_nn, "estimated_velocity") and policy_nn.estimated_velocity is not None:
                    estimated_velocity = policy_nn.estimated_velocity[0].detach().cpu()
                    estimated_velocity = estimated_velocity / estimator_target_scale

                    if "privileged" in obs_for_policy.keys():
                        true_velocity = obs_for_policy["privileged"][0].detach().cpu()
                        true_velocity = true_velocity / estimator_target_scale
                    else:
                        true_velocity = env.unwrapped.scene["robot"].data.root_lin_vel_b[0, :2].detach().cpu()

                    velocity_error = estimated_velocity - true_velocity
                    velocity_error_l2 = torch.linalg.vector_norm(velocity_error).item()

                    print("  Velocity estimator:")
                    print(f"    estimated base_lin_vel_xy: {estimated_velocity.numpy()}")
                    print(f"    true base_lin_vel_xy:      {true_velocity.numpy()}")
                    print(f"    error:                     {velocity_error.numpy()} |l2|={velocity_error_l2:.4f}")

                if env.unwrapped.num_envs == 1:
                    print("  Commands:")
                    for command_name in env.unwrapped.command_manager.active_terms:
                        try:
                            current_command = (
                                env.unwrapped.command_manager.get_command(command_name)[0].detach().cpu().numpy()
                            )
                        except Exception as exc:
                            print(f"    {command_name}: <unavailable: {exc}>")
                            continue
                        print(f"    {command_name}: {current_command}")
                    if args_cli.keyboard and keyboard_command_state["raw_command"] is not None:
                        raw_cmd = keyboard_command_state["raw_command"][0].detach().cpu().numpy()
                        desired_heading = keyboard_command_state["desired_heading"][0].item()
                        hold_yaw_command = keyboard_command_state["hold_yaw_command"][0].item()
                        print("  Keyboard heading-hold:")
                        print(f"    raw keyboard cmd: {raw_cmd}")
                        print(f"    desired heading:  {desired_heading:.4f}")
                        print(f"    hold yaw cmd:     {hold_yaw_command:.4f}")
                
                print(f"  Actions (synced):")
                print(f"    {'Idx':<5} {'Term->Joint':<35} {'Raw':<10} {'Scale':<8} {'Applied':<12} {'Offset/Note':<15}")
                print(f"    {'-'*80}")
                idx = 0
                ff_debug_terms = []
                for term_name, term in action_manager._terms.items():
                    processed = term.processed_actions[0].cpu().numpy()
                    offset = term._offset
                    for i, joint_name in enumerate(term._joint_names):
                        raw_val = action_values[idx]
                        applied_val = processed[i]
                        scale_val = term._scale if isinstance(term._scale, float) else term._scale[0, i].item()
                        
                        if hasattr(term, 'cfg') and hasattr(term.cfg, 'use_default_offset') and term.cfg.use_default_offset:
                            # Position control: target = raw * scale + default_pos
                            offset_val = offset if isinstance(offset, float) else offset[0, i].item()
                            note = f"offset={offset_val:.4f}"
                        else:
                            note = "no offset"
                        
                        print(f"    [{idx:<3}] {term_name}->{joint_name:<25} {raw_val:>8.4f} ×{scale_val:<6.2f} ={applied_val:>10.4f}  {note:<15}")
                        idx += 1
                    if hasattr(term, "ff_actions") and hasattr(term, "trigger_signal"):
                        ff_debug_terms.append((term_name, term))

                for term_name, term in ff_debug_terms:
                    lift_signal = term.lift_signal[0].detach().cpu().numpy()
                    ff_signal = term.ff_signal[0].detach().cpu().numpy()
                    trigger_signal = term.trigger_signal[0].detach().cpu().numpy()
                    ff_actions = term.ff_actions[0].detach().cpu().numpy()
                    ff_contribution = term.ff_contribution[0].detach().cpu().numpy()

                    print(f"  FF debug ({term_name}):")
                    print(f"    k_ff:            {term.k_ff:.4f}")
                    print(f"    lift_signal:     {lift_signal}")
                    print(f"    ff_signal:       {ff_signal}")
                    print(f"    trigger_signal:  {trigger_signal}")
                    print(f"    ff raw actions:")
                    for joint_name, ff_raw, ff_scaled in zip(term._joint_names, ff_actions, ff_contribution):
                        print(
                            f"      {joint_name:<25} raw={ff_raw:>8.4f}  contribution={ff_scaled:>8.4f}"
                        )
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
