# DDT_Lab Tita Guide

DDT_Lab is an Isaac Lab extension for training and evaluating DDT Tita locomotion
policies with RSL-RL. The current Tita tasks are organized around two policy
families:

- **MlpEstimator**: actor policy with an explicit velocity estimator.
- **DreamWaQ**: CENet/DreamWaQ-style context estimator with AdaBoot and constraint
  costs.

This README is written for the current code in this repository. Use the task
names listed below; older task names from previous experiments are not valid for
the current Tita registry.

## Supported Stack

Use the project with the following versions:

- Isaac Lab: `release/2.3.0`
- Isaac Sim: `5.1`
- Python: `3.11`
- RSL-RL: `rsl-rl-lib >= 3.0.1`

The training scripts launch Isaac Sim through Isaac Lab, so GPU access and a
working NVIDIA driver are required for normal CUDA training.

## Installation

Create and activate a Python environment:

```bash
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
python -m pip install --upgrade pip
```

Install Isaac Sim and Isaac Lab following the official Isaac Lab installation
flow. For this repository, use Isaac Lab `release/2.3.0` and install the RSL-RL
extras:

```bash
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
git checkout release/2.3.0
./isaaclab.sh --install rsl_rl
```

From this repository root, install DDT_Lab in editable mode:

```bash
cd <DDT_LAB_ROOT>
python -m pip install -e source/ddt_lab
```

Verify that the extension can be imported and the registered tasks can be listed:

```bash
python scripts/list_envs.py
```

## Tita Tasks

Training tasks should use names without `-Play-v0`. Play tasks are smaller
evaluation configurations and should be used for inference, visualization, and
export.

| Family | Terrain | Train task | Play task |
| --- | --- | --- | --- |
| MlpEstimator | Flat | `DDT-Velocity-Flat-Tita-MlpEstimator-v0` | `DDT-Velocity-Flat-Tita-MlpEstimator-Play-v0` |
| MlpEstimator | Rough | `DDT-Velocity-Rough-Tita-MlpEstimator-v0` | `DDT-Velocity-Rough-Tita-MlpEstimator-Play-v0` |
| MlpEstimator | Stair | `DDT-Velocity-Stair-Tita-MlpEstimator-v0` | `DDT-Velocity-Stair-Tita-MlpEstimator-Play-v0` |
| DreamWaQ | Flat | `DDT-Velocity-Flat-Tita-DreamWaQ-v0` | `DDT-Velocity-Flat-Tita-DreamWaQ-Play-v0` |
| DreamWaQ | Rough | `DDT-Velocity-Rough-Tita-DreamWaQ-v0` | `DDT-Velocity-Rough-Tita-DreamWaQ-Play-v0` |
| DreamWaQ | Stair | `DDT-Velocity-Stair-Tita-DreamWaQ-v0` | `DDT-Velocity-Stair-Tita-DreamWaQ-Play-v0` |

Default experiment folders:

| Train task group | Experiment folder |
| --- | --- |
| Flat MlpEstimator | `logs/rsl_rl/tita_flat_mlp_estimator` |
| Rough MlpEstimator | `logs/rsl_rl/tita_rough_mlp_estimator` |
| Stair MlpEstimator | `logs/rsl_rl/tita_stair_mlp_estimator` |
| Flat DreamWaQ | `logs/rsl_rl/tita_flat_dreamwaq_adaboot` |
| Rough DreamWaQ | `logs/rsl_rl/tita_rough_dreamwaq_adaboot` |
| Stair DreamWaQ | `logs/rsl_rl/tita_stair_dreamwaq_adaboot` |

## Training

Basic training:

```bash
python scripts/rsl_rl/train.py \
    --task DDT-Velocity-Stair-Tita-DreamWaQ-v0 \
    --headless \
    --seed 43 \
    --max_iterations 50000
```

Useful arguments:

- `--task`: registered task name.
- `--headless`: run without opening a GUI window.
- `--num_envs`: override the number of parallel environments.
- `--seed`: set the training seed. `--seed -1` samples a random seed.
- `--max_iterations`: override the runner's default iteration count.
- `--video`: record training videos. This also enables cameras.
- `--video_interval`: interval between video recordings.
- `--video_length`: number of steps per recorded video.
- `--run_name`: suffix added to the timestamped run directory.
- `--logger tensorboard|wandb|neptune`: choose the training logger.
- `--export_io_descriptors`: export manager-based environment I/O descriptors.

Record training video:

```bash
python scripts/rsl_rl/train.py \
    --task DDT-Velocity-Stair-Tita-DreamWaQ-v0 \
    --headless \
    --video \
    --video_interval 2000 \
    --video_length 200
```

Resume from a checkpoint in the task's experiment folder:

```bash
python scripts/rsl_rl/train.py \
    --task DDT-Velocity-Stair-Tita-DreamWaQ-v0 \
    --headless \
    --resume \
    --load_run 2026-06-01_18-53-38 \
    --checkpoint model_30000.pt
```

When resuming Tita feedforward-action tasks, the training script restores the
feedforward gain schedule from the checkpoint iteration when that iteration can
be resolved from the loaded checkpoint.

## Evaluation and Export

Run a trained policy with the matching Play task:

```bash
python scripts/rsl_rl/play.py \
    --task DDT-Velocity-Stair-Tita-DreamWaQ-Play-v0 \
    --checkpoint logs/rsl_rl/tita_stair_dreamwaq_adaboot/<run>/model_30000.pt \
    --num_envs 50
```

Headless evaluation with video:

```bash
python scripts/rsl_rl/play.py \
    --task DDT-Velocity-Stair-Tita-DreamWaQ-Play-v0 \
    --checkpoint logs/rsl_rl/tita_stair_dreamwaq_adaboot/<run>/model_30000.pt \
    --num_envs 50 \
    --headless \
    --video
```

Keyboard control for command debugging:

```bash
python scripts/rsl_rl/play.py \
    --task DDT-Velocity-Stair-Tita-DreamWaQ-Play-v0 \
    --checkpoint logs/rsl_rl/tita_stair_dreamwaq_adaboot/<run>/model_30000.pt \
    --num_envs 1 \
    --keyboard \
    --real-time
```

Feedforward gain override in play mode:

```bash
python scripts/rsl_rl/play.py \
    --task DDT-Velocity-Stair-Tita-DreamWaQ-Play-v0 \
    --checkpoint logs/rsl_rl/tita_stair_dreamwaq_adaboot/<run>/model_30000.pt \
    --kff 0
```

`--kff 0` disables the joint-position feedforward term during play. Use a
positive value to test a fixed feedforward gain.

The play script exports deployment artifacts to:

```text
logs/rsl_rl/<experiment_name>/<run>/exported
```

Common exported files:

- Standard policies: `policy.pt`, `policy.onnx`.
- MlpEstimator policies: `policy.pt`, `policy.onnx`, `policy_metadata.json`,
  `estimator_policy.pt`, `estimator_policy.onnx`, `actor_policy.pt`,
  `actor_policy.onnx`, `policy_split_metadata.json`.
- DreamWaQ policies: `policy.pt`, `policy.onnx`, `policy_metadata.json`,
  split CENet encoder/head/actor files, and `policy_split_metadata.json`.

The ONNX opset defaults to `15` and can be changed with:

```bash
python scripts/rsl_rl/play.py \
    --task DDT-Velocity-Stair-Tita-DreamWaQ-Play-v0 \
    --checkpoint logs/rsl_rl/tita_stair_dreamwaq_adaboot/<run>/model_30000.pt \
    --onnx_opset 18
```

## Logs

Training runs are written to:

```text
logs/rsl_rl/<experiment_name>/<timestamp>[_<run_name>]
```

Important files and folders:

- `params/env.yaml`: resolved environment configuration.
- `params/agent.yaml`: resolved RSL-RL runner configuration.
- `model_*.pt`: checkpoints.
- `videos/train`: training videos when `--video` is enabled.
- `videos/play`: play videos when `--video` is enabled.
- `exported`: TorchScript, ONNX, and metadata files written by `play.py`.
- `git`: captured repository diff for reproducibility.

Use TensorBoard to inspect training curves:

```bash
tensorboard --logdir ./logs
```

Then open:

```text
http://localhost:6006
```

## Dummy Agents

Use dummy agents to sanity-check an environment without training a policy.

Zero-action agent:

```bash
python scripts/zero_agent.py --task DDT-Velocity-Stair-Tita-DreamWaQ-v0
```

Random-action agent:

```bash
python scripts/random_agent.py --task DDT-Velocity-Stair-Tita-DreamWaQ-v0
```

## Diagnostics and Troubleshooting

`[diag]` messages are diagnostic prints from the local observation, action, and
RSL-RL extension code. They are not always failures.

- `[diag][obs_term] ... invalid=0`: the tensor is finite; the logger is reporting
  a large observation value above a configured threshold.
- `[diag][safe_height_scan] repaired ...`: invalid height-scan rays were repaired
  before entering the policy.
- `[diag][tita_action] ...`: reports feedforward and policy action blending.

Treat diagnostics as serious when they show persistent non-finite values,
exploding rewards/losses, or immediate environment failure.

Common issues:

- `RuntimeError: No CUDA GPUs are available`: the process cannot access the GPU.
  Check the NVIDIA driver, container GPU permissions, or sandbox settings.
- `Failed to open the default display`: use `--headless` on servers without a
  graphical display.
- `Not all regular expressions are matched`: a joint or body regex in the config
  does not match the robot asset. For current Tita wheel joints, configs should
  match `.*_leg_4`.
- Checkpoint not found: use the task's experiment folder and pass either a full
  checkpoint path with `--checkpoint`, or use `--load_run <run>` with
  `--checkpoint model_<iteration>.pt`.

## Real-Robot Deployment

For sim-to-sim and sim-to-real deployment workflows, see:

```text
https://github.com/DDTRobot/tita_rl_sim2sim2real
```

## Developer Notes

Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit run --all-files
```

For IDE indexing, add the extension package to your Python analysis paths:

```json
{
    "python.analysis.extraPaths": [
        "<DDT_LAB_ROOT>/source/ddt_lab"
    ]
}
```

If Isaac Sim packages overwhelm the language server, exclude unused Omniverse
extension caches from indexing in your editor settings.
