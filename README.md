# Predictive Representation Learning for Policy-Gradient CTDE

This repository contains the implementation of Predictive Representation Learning (PRL) for cooperative multi-agent reinforcement learning under the centralized-training decentralized-execution (CTDE) paradigm.

The codebase is built upon [PKU-MARL/HARL](https://github.com/PKU-MARL/HARL), the official PyTorch implementation of Heterogeneous-Agent Reinforcement Learning algorithms, including HAPPO, HATRPO, MAPPO, and related baselines. We extend HARL with actor-side predictive representation learning modules, additional controlled baselines, experiment configurations, probing utilities, and plotting scripts.


## Overview

PRL uses the centralized critic as a training-time representation teacher. During training, each decentralized actor is additionally supervised to predict the critic-side representation of the next global state from its own local observation. This auxiliary objective is combined with the original on-policy actor objective and preserves decentralized execution.

This repository includes:

- PRL extensions for policy-gradient CTDE algorithms.
- Capacity-matched deep actor baselines.
- Alternative actor-side prediction targets, including current-state prediction and teammate-action prediction.
- Experiment configurations for GRF, MA-MuJoCo, and Bi-DexHands.
- Linear probing utilities for analyzing actor representations.
- Plotting and table-generation scripts used for paper figures and numerical summaries.

## Repository Structure

```text
harl/
  algorithms/              # On-policy algorithm implementations and PRL-related modifications
  models/                  # Actor, critic, MLP, and deep residual network modules
  runners/                 # Training runners and auxiliary-loss integration
  common/                  # Buffers, utilities, and shared training components
  probing_tools/                   # Probing and analysis utilities

examples/
  train.py                 # Main training entry point

tuned_configs/
  grf/                     # Google Research Football configs
  mamujoco/                # Multi-Agent MuJoCo configs
  dexhands/                # Bi-DexHands configs

scripts/
  *.sh                     # Example launch scripts

results/
  ...                      # Training logs, checkpoints, and TensorBoard files

plotting/
  ...                      # Plotting and table-generation utilities
```

Some files may contain intermediate experimental utilities developed during the research process. The core training and evaluation logic is contained in `harl/`, `examples/train.py`, `tuned_configs/`, and the corresponding plotting/probing scripts.

## Installation

Create a conda environment:

```bash
conda create -n marl_prl python=3.8
conda activate marl_prl
```

Install PyTorch according to your CUDA version. For example:

```bash
pip install torch
```

Install the repository in editable mode:

```bash
pip install -e .
```

Additional environment dependencies depend on the benchmark used. Please follow the original HARL installation instructions for Google Research Football, Multi-Agent MuJoCo, Bi-DexHands, SMAC, SMACv2, MPE, and other supported environments.

## Training

A training run can be launched through `examples/train.py` with a configuration file:

```bash
python examples/train.py \
  --load_config tuned_configs/<env>/<task>/<method>/config.json \
  --seed 1 \
  --log_dir results/<experiment_name>_seed_1
```

Example:

```bash
python examples/train.py \
  --load_config tuned_configs/grf/academy_3_vs_1_with_keeper/prl/config.json \
  --seed 1 \
  --log_dir results/grf_3_vs_1_with_keeper_seed_1
```

The generated logs, checkpoints, TensorBoard files, and saved configurations are stored under the specified `results/` directory.

## Evaluation and Plotting

Training curves and final performance tables can be generated using the plotting scripts included in this repository. Typical usage:

```bash
python plot_multitask.py \
  --dirs results/task_seed_1 results/task_seed_2 results/task_seed_3 \
  --task_names task_name \
  --task_tags eval_average_episode_rewards \
  --include PRL_ours HAPPO_baseline HAPPO_deep_baseline MAPPO_baseline \
  --save_dir plots
```

The exact command depends on the tasks, seeds, and methods included in the experiment.

## Representation Probing

Actor representations can be analyzed using frozen-encoder linear probes. A typical command is:

```bash
python harl/probing_tools/run_probe_by_task.py \
  --task grf_3_vs_1_with_keeper \
  --seed 1 \
  --methods HAPPO_baseline,HAPPO_deep_baseline,PRL_ours \
  --stage final \
  --checkpoint backup_ep2500.pt \
  --rollout_steps 20000 \
  --train_epochs 50 \
  --batch_size 1024 \
  --lr 1e-3 \
  --probe_type linear \
  --device cuda
```

The probing scripts are used to evaluate whether actor encoders contain information about global state, teammate actions, and critic-side predictive targets.

## Notes on Reproducibility

The experimental code follows the HARL training interface and configuration style. Most experiments are controlled through JSON configuration files under `tuned_configs/`.

Important implementation details include:

- PPO-style clipped policy updates.
- Generalized Advantage Estimation.
- Value normalization.
- Actor-side auxiliary prediction loss for PRL.
- Deep residual actor backbone for PRL and capacity-matched deep baselines.
- Task-specific PRL loss coefficients.

Please refer to the corresponding configuration files for exact hyperparameters.

## Relationship to HARL

This repository is a modified research codebase based on PKU-MARL/HARL. The original HARL components remain under their original license and attribution. Our modifications add PRL-specific training objectives, model components, configurations, analysis tools, and plotting utilities.

Original HARL repository:

```text
https://github.com/PKU-MARL/HARL
```

If you use the original HARL components, please cite the HARL paper as requested by the original authors. If you use the PRL extensions in this repository, please cite our paper once available.

## Citation

For HARL, please cite:

```bibtex
@article{JMLR:v25:23-0488,
  author  = {Yifan Zhong and Jakub Grudzien Kuba and Xidong Feng and Siyi Hu and Jiaming Ji and Yaodong Yang},
  title   = {Heterogeneous-Agent Reinforcement Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {32},
  pages   = {1--67},
  url     = {http://jmlr.org/papers/v25/23-0488.html}
}
```

For this repository, please cite the corresponding PRL paper once released.

## License and Attribution

This repository is derived from HARL, originally developed by PKU-MARL. HARL declares the MIT License in its `setup.py`; we preserve that license and release our PRL modifications under the same MIT License.

The benchmark environments used in the paper include Google Research Football, Multi-Agent MuJoCo, and Bi-DexHands. Please refer to their respective repositories for their licenses and terms of use.
