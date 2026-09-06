
"""
Probe experiment for actor embeddings in HARL_PRED_NEXT.

Goal:
  For each method checkpoint, roll out with that method's own policy, collect each agent's
  actor embedding h_t^i, and train probes for three uncertainty-related targets:

  1) partial observation uncertainty: predict current centralized state s_t from h_t^i
  2) teammate action uncertainty: predict other agents' actions a_t^{-i} from h_t^i
  3) transition uncertainty: predict next critic latent z_{t+1}=g_phi(s_{t+1}) from h_t^i

Manifest format supports both:

A) Dict format:
{
  "runs": [
    {
      "method": "PRL",
      "stage": "final",
      "config": "tuned_configs/...",
      "model_dir": "results/.../models",
      "checkpoint": "backup_ep2500.pt"
    }
  ]
}

B) List format:
[
  {
    "method": "PRL",
    "stage": "final",
    "config": "tuned_configs/...",
    "model_dir": "results/.../models",
    "checkpoint": "backup_ep2500.pt"
  }
]

Notes:
  - This script is evaluation-only. It does not call runner.train().
  - It uses each checkpoint's own policy to generate its own probe dataset.
  - If "checkpoint" is provided, it loads that checkpoint rather than latest.pt.
  - If "checkpoint" is omitted, it loads latest.pt from model_dir.
"""

from __future__ import annotations

try:
    import isaacgym  
except ImportError:
    pass

import argparse
import copy
import csv
import json
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.utils.trans_tools import _t2n




def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ProbeResult:
    method: str
    stage: str
    agent_id: int
    target_name: str
    probe_type: str
    n_train: int
    n_val: int
    mse: float
    target_var: float
    norm_mse: float
    r2: float



def resolve_checkpoint_path(model_dir: str, checkpoint: Optional[str]) -> str:
    """
    Resolve checkpoint path.

    If checkpoint is None:
      return model_dir/latest.pt

    If checkpoint is relative:
      return model_dir/checkpoint

    If checkpoint is absolute:
      return checkpoint
    """
    model_dir = str(model_dir)

    if checkpoint is None:
        ckpt_path = os.path.join(model_dir, "latest.pt")
    elif os.path.isabs(checkpoint):
        ckpt_path = checkpoint
    else:
        ckpt_path = os.path.join(model_dir, checkpoint)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    return ckpt_path


def make_temp_model_dir_for_checkpoint(
    model_dir: str,
    checkpoint: Optional[str],
    tmp_root: str,
) -> str:
    """
    OnPolicyBaseRunner.restore() always tries model_dir/latest.pt first.

    To load a specific backup checkpoint without changing core HARL code,
    we create a temporary model directory and copy/symlink the selected checkpoint
    as latest.pt.

    This is intentionally probe-only and does not affect original training files.
    """
    model_dir = str(model_dir)


    if checkpoint is None:
        latest_path = os.path.join(model_dir, "latest.pt")
        if not os.path.exists(latest_path):
            raise FileNotFoundError(f"latest.pt not found in model_dir: {latest_path}")
        return model_dir

    ckpt_path = resolve_checkpoint_path(model_dir, checkpoint)

    tmp_model_dir = tempfile.mkdtemp(prefix="probe_ckpt_", dir=tmp_root)
    tmp_latest = os.path.join(tmp_model_dir, "latest.pt")

    try:
        os.symlink(os.path.abspath(ckpt_path), tmp_latest)
    except OSError:
        shutil.copy2(ckpt_path, tmp_latest)

    return tmp_model_dir


def load_config(
    config_path: str,
    effective_model_dir: str,
    seed: Optional[int] = None,
) -> Tuple[dict, dict, dict]:
    with open(config_path, "r") as f:
        cfg = json.load(f)

    main_args = copy.deepcopy(cfg["main_args"])
    algo_args = copy.deepcopy(cfg["algo_args"])
    env_args = copy.deepcopy(cfg["env_args"])

   
    algo_args["train"]["model_dir"] = effective_model_dir
    algo_args["render"]["use_render"] = False
    algo_args["eval"]["use_eval"] = False

   
    algo_args["train"]["num_env_steps"] = max(
        algo_args["train"]["episode_length"] * algo_args["train"]["n_rollout_threads"],
        algo_args["train"].get("num_env_steps", 1),
    )

    if seed is not None:
        algo_args["seed"]["seed"] = seed
        algo_args["seed"]["seed_specify"] = True

    return main_args, algo_args, env_args


def build_runner(
    config_path: str,
    model_dir: str,
    checkpoint: Optional[str],
    tmp_root: str,
    seed: Optional[int] = None,
) -> OnPolicyHARunner:
    effective_model_dir = make_temp_model_dir_for_checkpoint(
        model_dir=model_dir,
        checkpoint=checkpoint,
        tmp_root=tmp_root,
    )

    main_args, algo_args, env_args = load_config(
        config_path=config_path,
        effective_model_dir=effective_model_dir,
        seed=seed,
    )

    runner = OnPolicyHARunner(main_args, algo_args, env_args)
    runner.prep_rollout()
    return runner




@torch.no_grad()
def actor_embedding_and_action(
    runner: OnPolicyHARunner,
    agent_id: int,
    obs_np: np.ndarray,
    rnn_state_np: np.ndarray,
    mask_np: np.ndarray,
    available_actions_np: Optional[np.ndarray],
    deterministic: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      h_t^i, action, next actor rnn state for one agent.

    h_t^i is the actor representation used by the action distribution.
    For recurrent policies, this is the post-RNN feature.
    """
    actor_algo = runner.actor[agent_id]
    policy = actor_algo.actor
    device = actor_algo.device
    dtype = torch.float32

    obs = torch.as_tensor(obs_np, dtype=dtype, device=device)
    rnn_state = torch.as_tensor(rnn_state_np, dtype=dtype, device=device)
    mask = torch.as_tensor(mask_np, dtype=dtype, device=device)

    available_actions = None
    if available_actions_np is not None:
        available_actions = torch.as_tensor(
            available_actions_np,
            dtype=dtype,
            device=device,
        )

    features = policy.base(obs)

    if policy.use_naive_recurrent_policy or policy.use_recurrent_policy:
        features, new_rnn_state = policy.rnn(features, rnn_state, mask)
    else:
        new_rnn_state = rnn_state

    actions, _ = policy.act(features, available_actions, deterministic=deterministic)

    return _t2n(features), _t2n(actions), _t2n(new_rnn_state)


@torch.no_grad()
def critic_next_embedding(
    runner: OnPolicyHARunner,
    next_share_obs_np: np.ndarray,
) -> np.ndarray:
    """
    Compute detached critic encoder target:
      z_{t+1} = g_phi(s_{t+1})
    """
    b_size = next_share_obs_np.shape[0]
    rec_n = runner.critic.critic.recurrent_n
    h_size = runner.critic.critic.hidden_sizes[-1]

    dummy_rnn = np.zeros((b_size, rec_n, h_size), dtype=np.float32)
    dummy_masks = np.ones((b_size, 1), dtype=np.float32)

    z = runner.critic.critic.get_embedding(
        next_share_obs_np,
        dummy_rnn,
        dummy_masks,
    )
    return _t2n(z)



def collect_probe_dataset(
    runner: OnPolicyHARunner,
    rollout_steps: int,
    deterministic: bool = False,
    desc: str = "Collect rollout",
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Collect per-agent probe data using the checkpoint's own policy.
    """
    n_threads = runner.algo_args["train"]["n_rollout_threads"]
    num_agents = runner.num_agents
    hidden_size = runner.rnn_hidden_size
    recurrent_n = runner.recurrent_n

    obs, share_obs, available_actions = runner.envs.reset()

    rnn_states = np.zeros(
        (n_threads, num_agents, recurrent_n, hidden_size),
        dtype=np.float32,
    )
    masks = np.ones((n_threads, num_agents, 1), dtype=np.float32)

    store: Dict[int, Dict[str, List[np.ndarray]]] = {
        i: {
            "h": [],
            "state": [],
            "next_state": [],
            "others_action": [],
            "next_z": [],
        }
        for i in range(num_agents)
    }

    steps_done = 0

    pbar = tqdm(
        total=rollout_steps,
        desc=desc,
        unit="env-step",
        dynamic_ncols=True,
        leave=True,
    )

    while steps_done < rollout_steps:
        h_list: List[np.ndarray] = []
        action_list: List[np.ndarray] = []
        new_rnn_list: List[np.ndarray] = []

        for agent_id in range(num_agents):
            avail_i = None
            if available_actions is not None and available_actions[0] is not None:
                avail_i = available_actions[:, agent_id]

            h_i, a_i, new_rnn_i = actor_embedding_and_action(
                runner=runner,
                agent_id=agent_id,
                obs_np=obs[:, agent_id],
                rnn_state_np=rnn_states[:, agent_id],
                mask_np=masks[:, agent_id],
                available_actions_np=avail_i,
                deterministic=deterministic,
            )

            h_list.append(h_i)
            action_list.append(a_i)
            new_rnn_list.append(new_rnn_i)

        actions = np.stack(action_list, axis=1) 
        h_all = np.stack(h_list, axis=1)        

        next_obs, next_share_obs, rewards, dones, infos, next_available_actions = (
            runner.envs.step(actions)
        )

        if runner.state_type == "EP":
            state_t = share_obs[:, 0]
            state_tp1 = next_share_obs[:, 0]
        elif runner.state_type == "FP":
            state_t = share_obs.reshape(share_obs.shape[0], -1)
            state_tp1 = next_share_obs.reshape(next_share_obs.shape[0], -1)
        else:
            raise NotImplementedError(f"Unknown state_type: {runner.state_type}")

        z_tp1 = critic_next_embedding(runner, state_tp1)

        for agent_id in range(num_agents):
            other_actions = [
                actions[:, j]
                for j in range(num_agents)
                if j != agent_id
            ]
            other_actions = np.concatenate(other_actions, axis=-1)

            store[agent_id]["h"].append(h_all[:, agent_id])
            store[agent_id]["state"].append(state_t)
            store[agent_id]["next_state"].append(state_tp1)
            store[agent_id]["others_action"].append(other_actions)
            store[agent_id]["next_z"].append(z_tp1)

        dones_env = np.all(dones, axis=1)

        rnn_states = np.stack(new_rnn_list, axis=1)
        rnn_states[dones_env] = 0.0

        masks = np.ones((n_threads, num_agents, 1), dtype=np.float32)
        masks[dones_env] = 0.0

        obs = next_obs
        share_obs = next_share_obs
        available_actions = next_available_actions

        step_inc = min(n_threads, rollout_steps - steps_done)
        steps_done += n_threads
        pbar.update(step_inc)

    pbar.close()

    out: Dict[int, Dict[str, np.ndarray]] = {}

    for agent_id in range(num_agents):
        out[agent_id] = {
            key: np.concatenate(vals, axis=0).astype(np.float32)
            for key, vals in store[agent_id].items()
        }

    return out




def standardize_train_val(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_mean = x_train.mean(0, keepdim=True)
    x_std = x_train.std(0, keepdim=True).clamp_min(eps)

    y_mean = y_train.mean(0, keepdim=True)
    y_std = y_train.std(0, keepdim=True).clamp_min(eps)

    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std

    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std

    return x_train, y_train, x_val, y_val


def train_one_probe(
    x_np: np.ndarray,
    y_np: np.ndarray,
    probe_type: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
    desc: str,
) -> Tuple[float, float, float, float, int, int]:
    set_global_seed(seed)

    x = torch.as_tensor(x_np, dtype=torch.float32)
    y = torch.as_tensor(y_np, dtype=torch.float32)

    n = x.shape[0]
    n_train = int(0.8 * n)
    n_val = n - n_train

    perm = torch.randperm(n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    x_train_raw = x[train_idx]
    y_train_raw = y[train_idx]
    x_val_raw = x[val_idx]
    y_val_raw = y[val_idx]

    x_train, y_train, x_val, y_val = standardize_train_val(
        x_train_raw,
        y_train_raw,
        x_val_raw,
        y_val_raw,
    )

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    in_dim = x.shape[1]
    out_dim = y.shape[1]

    if probe_type == "linear":
        model = LinearProbe(in_dim, out_dim)
    elif probe_type == "mlp":
        model = MLPProbe(in_dim, out_dim)
    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")

    model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    x_val = x_val.to(device)
    y_val = y_val.to(device)

    epoch_bar = tqdm(
        range(epochs),
        desc=desc,
        unit="epoch",
        dynamic_ncols=True,
        leave=False,
    )

    for _ in epoch_bar:
        model.train()

        running_loss = 0.0
        n_batches = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = F.mse_loss(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            opt.step()

            running_loss += loss.item()
            n_batches += 1

        if n_batches > 0:
            epoch_bar.set_postfix(loss=f"{running_loss / n_batches:.4f}")

    model.eval()

    with torch.no_grad():
        pred_val = model(x_val)
        mse = F.mse_loss(pred_val, y_val).item()
        target_var = torch.var(y_val, unbiased=False).item()
        norm_mse = mse / max(target_var, 1e-8)
        r2 = 1.0 - norm_mse

    return mse, target_var, norm_mse, r2, n_train, n_val


def run_probes_for_dataset(
    method: str,
    stage: str,
    dataset: Dict[int, Dict[str, np.ndarray]],
    probe_types: List[str],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> List[ProbeResult]:
    results: List[ProbeResult] = []

    target_map = {
        "partial_observation_state": "state",
        "next_state": "next_state",
        "teammate_action": "others_action",
        "transition_next_critic_latent": "next_z",
    }

    total_jobs = len(dataset) * len(target_map) * len(probe_types)

    job_bar = tqdm(
        total=total_jobs,
        desc=f"Train probes [{method}-{stage}]",
        unit="probe",
        dynamic_ncols=True,
        leave=True,
    )

    for agent_id, data in dataset.items():
        x = data["h"]

        for target_name, key in target_map.items():
            y = data[key]

            for probe_type in probe_types:
                desc = (
                    f"{method}-{stage} | "
                    f"agent={agent_id} | "
                    f"{target_name} | "
                    f"{probe_type}"
                )

                mse, target_var, norm_mse, r2, n_train, n_val = train_one_probe(
                    x_np=x,
                    y_np=y,
                    probe_type=probe_type,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    device=device,
                    seed=seed + agent_id,
                    desc=desc,
                )

                results.append(
                    ProbeResult(
                        method=method,
                        stage=stage,
                        agent_id=agent_id,
                        target_name=target_name,
                        probe_type=probe_type,
                        n_train=n_train,
                        n_val=n_val,
                        mse=mse,
                        target_var=target_var,
                        norm_mse=norm_mse,
                        r2=r2,
                    )
                )

                job_bar.set_postfix(
                    agent=agent_id,
                    target=target_name,
                    probe=probe_type,
                    r2=f"{r2:.4f}",
                )
                job_bar.update(1)

    job_bar.close()

    return results




def write_results_csv(results: List[ProbeResult], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "stage",
                "agent_id",
                "target_name",
                "probe_type",
                "n_train",
                "n_val",
                "mse",
                "target_var",
                "norm_mse",
                "r2",
            ],
        )

        writer.writeheader()

        for r in results:
            writer.writerow(r.__dict__)


def save_npz_dataset(
    dataset: Dict[int, Dict[str, np.ndarray]],
    out_path: str,
) -> None:
    arrays = {}

    for agent_id, data in dataset.items():
        for key, val in data.items():
            arrays[f"agent{agent_id}_{key}"] = val

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)


def normalize_manifest(manifest_obj) -> List[dict]:
    """
    Support both:
      {"runs": [...]}
    and:
      [...]
    """
    if isinstance(manifest_obj, dict):
        if "runs" not in manifest_obj:
            raise KeyError("Manifest dict must contain key: 'runs'")
        runs = manifest_obj["runs"]
    elif isinstance(manifest_obj, list):
        runs = manifest_obj
    else:
        raise TypeError("Manifest must be either a dict with key 'runs' or a list.")

    if not isinstance(runs, list):
        raise TypeError("Manifest runs must be a list.")

    required_keys = ["method", "stage", "config", "model_dir"]

    for idx, run in enumerate(runs):
        for key in required_keys:
            if key not in run:
                raise KeyError(f"Run {idx} missing required key: {key}")

    return runs




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--rollout_steps", type=int, default=20000)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument(
        "--probe_types",
        nargs="+",
        default=["linear"],
        choices=["linear", "mlp"],
    )

    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument("--save_datasets", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    with open(args.manifest, "r") as f:
        manifest_obj = json.load(f)

    runs = normalize_manifest(manifest_obj)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = out_dir / "_tmp_model_dirs"
    tmp_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    all_results: List[ProbeResult] = []

    run_bar = tqdm(
        runs,
        desc="Probe runs",
        unit="run",
        dynamic_ncols=True,
        leave=True,
    )

    for run in run_bar:
        method = run["method"]
        stage = run["stage"]
        config_path = run["config"]
        model_dir = run["model_dir"]
        checkpoint = run.get("checkpoint", None)

        run_bar.set_postfix(method=method, stage=stage)

        print("\n" + "=" * 80)
        print(f"Collecting dataset")
        print(f"method     : {method}")
        print(f"stage      : {stage}")
        print(f"config     : {config_path}")
        print(f"model_dir  : {model_dir}")
        print(f"checkpoint : {checkpoint if checkpoint is not None else 'latest.pt'}")
        print("=" * 80)

        runner = build_runner(
            config_path=config_path,
            model_dir=model_dir,
            checkpoint=checkpoint,
            tmp_root=str(tmp_root),
            seed=args.seed,
        )

        try:
            dataset = collect_probe_dataset(
                runner=runner,
                rollout_steps=args.rollout_steps,
                deterministic=args.deterministic,
                desc=f"Rollout [{method}-{stage}]",
            )
        finally:
            runner.close()

        if args.save_datasets:
            ds_path = out_dir / "datasets" / f"{method}_{stage}.npz"
            save_npz_dataset(dataset, str(ds_path))
            print(f"Saved dataset: {ds_path}")

        print("\n" + "-" * 80)
        print(f"Training probes: method={method}, stage={stage}")
        print("-" * 80)

        results = run_probes_for_dataset(
            method=method,
            stage=stage,
            dataset=dataset,
            probe_types=args.probe_types,
            epochs=args.train_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            seed=args.seed,
        )

        all_results.extend(results)

        partial_csv = out_dir / f"probe_results_{method}_{stage}.csv"
        write_results_csv(results, str(partial_csv))
        print(f"Wrote partial results: {partial_csv}")

    final_csv = out_dir / "probe_results_all.csv"
    write_results_csv(all_results, str(final_csv))

    print("\n" + "=" * 80)
    print(f"Done. Wrote final results: {final_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()