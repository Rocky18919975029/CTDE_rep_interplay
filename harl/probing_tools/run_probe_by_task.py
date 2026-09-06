
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt



TASK_SPECS: Dict[str, dict] = {
    "mamujoco_humanoid_17x1": {
        "result_root": "results/mamujoco_humanoid_17x1_seed_{seed}/mamujoco/Humanoid-v2-17x1",
        "stage_checkpoints": {
            "early": "backup_ep250.pt",
            "middle": "backup_ep1250.pt",
            "final": "backup_ep2500.pt",
        },
    },

  
    "grf_counterattack_hard": {
        "result_root": "results/grf_counterattack_hard_seed_{seed}/football/academy_counterattack_hard",
      
        "stage_checkpoints": {
            "early": "backup_ep200.pt",    
            "middle": "backup_ep500.pt",  
            "final": "backup_ep1000.pt",  
        },
    },
    "dexhands_SHO": {
        "result_root": "results/dexhands_SHO_seed_{seed}/dexhands/ShadowHandOver",
       
        "stage_checkpoints": {
            "early": "backup_ep520.pt",     
            "middle": "backup_ep1300.pt",  
            "final": "backup_ep2600.pt",   
        },
    },
       
    "grf_3_vs_1_with_keeper": {
        "result_root": "results/grf_3_vs_1_with_keeper_seed_{seed}/football/academy_3_vs_1_with_keeper",
        
        "stage_checkpoints": {
            "early": "backup_ep250.pt",     
            "middle": "backup_ep1250.pt",   
            "final": "backup_ep2500.pt",   
        },
    },
}


TARGET_DISPLAY = {
    "partial_observation_state": "Partial Obs. Uncertainty",
    "teammate_action": "Teammate Action Uncertainty",
    "transition_next_critic_latent": "Transition Uncertainty",
}


METHOD_DISPLAY = {
    "HAPPO_baseline": "HAPPO",
    "HAPPO_deep_baseline": "HAPPO_deep",
    "MAPPO_baseline": "MAPPO",
    "PRL_without_Norm": "PRL",
    "PRL_ours": "PRL",
}



def parse_methods(methods: str) -> List[str]:
    return [m.strip() for m in methods.split(",") if m.strip()]


def find_model_dir(task: str, seed: int, method: str) -> Path:
    if task not in TASK_SPECS:
        raise ValueError(f"Unknown task: {task}. Available: {list(TASK_SPECS.keys())}")

    root = Path(TASK_SPECS[task]["result_root"].format(seed=seed))

    if not root.exists():
        raise FileNotFoundError(f"Task result root not found: {root}")

    candidates = list(root.glob(f"*/{method}/seed-*/models"))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No model dir found for method={method} under:\n{root}\n"
            f"Expected pattern: */{method}/seed-*/models"
        )

    if len(candidates) > 1:
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"[Warning] Multiple candidates found for {method}. Using latest modified:")
        for p in candidates:
            print(f"  - {p}")

    return candidates[0]


def infer_config_path(model_dir: Path) -> Path:
    config_path = model_dir.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json not found next to models dir:\n{config_path}"
        )
    return config_path

def make_probe_config(original_config_path: Path, out_dir: Path, method: str, stage: str) -> Path:
    """
    Create a probe-only config to avoid polluting the original training results directory.

    Key change:
    - Redirect algo_args.logger.log_dir to the probe output directory.
    - Keep all other training / model / env parameters unchanged.
    """
    with open(original_config_path, "r") as f:
        cfg = json.load(f)

    probe_log_dir = out_dir / "_probe_logs" / method / stage
    probe_log_dir.mkdir(parents=True, exist_ok=True)

    cfg.setdefault("algo_args", {})
    cfg["algo_args"].setdefault("logger", {})
    cfg["algo_args"]["logger"]["log_dir"] = str(probe_log_dir)


    probe_config_dir = out_dir / "_probe_configs"
    probe_config_dir.mkdir(parents=True, exist_ok=True)

    probe_config_path = probe_config_dir / f"{method}_{stage}_config.json"
    with open(probe_config_path, "w") as f:
        json.dump(cfg, f, indent=4)

    return probe_config_path


def resolve_checkpoint(task: str, stage: str, model_dir: Path, checkpoint_override: Optional[str]) -> str:
    if checkpoint_override is not None:
        return checkpoint_override

    stage_map = TASK_SPECS[task]["stage_checkpoints"]

    if stage not in stage_map:
        raise ValueError(
            f"Unknown stage={stage}. Available stages: {list(stage_map.keys())}"
        )

    checkpoint = stage_map[stage]
    ckpt_path = model_dir / checkpoint

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{ckpt_path}\n"
            f"Use --checkpoint to manually specify another checkpoint."
        )

    return checkpoint


def build_manifest(
    task: str,
    seed: int,
    methods: List[str],
    stage: str,
    checkpoint_override: Optional[str],
    manifest_path: Path,
) -> Path:
    runs = []

    for method in methods:
        model_dir = find_model_dir(task, seed, method)
        original_config_path = infer_config_path(model_dir)

        config_path = make_probe_config(
            original_config_path=original_config_path,
            out_dir=manifest_path.parent,
            method=method,
            stage=stage,
        )
  

        checkpoint = resolve_checkpoint(task, stage, model_dir, checkpoint_override)

        run = {
            "method": METHOD_DISPLAY.get(method, method),
            "raw_method": method,
            "stage": stage,
            "config": str(config_path),
            "model_dir": str(model_dir),
            "checkpoint": checkpoint,
        }
        runs.append(run)

    manifest = {"runs": runs}

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"\nWrote manifest: {manifest_path}")
    for r in runs:
        print(
            f"  method={r['method']} | raw_method={r['raw_method']} | "
            f"stage={r['stage']} | checkpoint={r['checkpoint']}"
        )

    return manifest_path




def aggregate_results(csv_path: Path, probe_type: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "probe_type" in df.columns:
        df = df[df["probe_type"] == probe_type].copy()

    df["target_display"] = df["target_name"].map(TARGET_DISPLAY).fillna(df["target_name"])

    agg = (
        df.groupby(["method", "stage", "target_display"], as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            norm_mse_mean=("norm_mse", "mean"),
            norm_mse_std=("norm_mse", "std"),
        )
    )

    agg["r2_std"] = agg["r2_std"].fillna(0.0)
    agg["norm_mse_std"] = agg["norm_mse_std"].fillna(0.0)

    return agg


def plot_metric(
    agg: pd.DataFrame,
    stage: str,
    metric_mean: str,
    metric_std: str,
    ylabel: str,
    title: str,
    save_path: Path,
) -> None:
    targets = [
        "Partial Obs. Uncertainty",
        "Teammate Action Uncertainty",
        "Transition Uncertainty",
    ]

    methods = list(agg["method"].drop_duplicates())

    x = range(len(targets))
    width = 0.8 / max(len(methods), 1)

    plt.figure(figsize=(14, 7))

    for i, method in enumerate(methods):
        vals = []
        errs = []

        for target in targets:
            row = agg[(agg["method"] == method) & (agg["target_display"] == target)]
            if len(row) == 0:
                vals.append(0.0)
                errs.append(0.0)
            else:
                vals.append(float(row[metric_mean].iloc[0]))
                errs.append(float(row[metric_std].iloc[0]))

        xpos = [v + (i - (len(methods) - 1) / 2) * width for v in x]
        plt.bar(xpos, vals, width=width, label=method, yerr=errs, capsize=2)

    plt.xticks(list(x), targets, rotation=15, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Wrote plot: {save_path}")


def make_plots(out_dir: Path, stage: str, probe_type: str) -> None:
    csv_path = out_dir / "probe_results_all.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Probe result CSV not found: {csv_path}")

    agg = aggregate_results(csv_path, probe_type=probe_type)

    agg_csv = out_dir / f"{stage}_{probe_type}_summary.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"Wrote summary CSV: {agg_csv}")

    plot_metric(
        agg=agg,
        stage=stage,
        metric_mean="r2_mean",
        metric_std="r2_std",
        ylabel="R^2 (higher is better)",
        title=f"{stage.capitalize()}-stage Probe Performance (R^2)",
        save_path=out_dir / f"{stage}_{probe_type}_r2_bar.png",
    )

    plot_metric(
        agg=agg,
        stage=stage,
        metric_mean="norm_mse_mean",
        metric_std="norm_mse_std",
        ylabel="Normalized MSE (lower is better)",
        title=f"{stage.capitalize()}-stage Probe Performance (Normalized MSE)",
        save_path=out_dir / f"{stage}_{probe_type}_norm_mse_bar.png",
    )




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--methods",
        type=str,
        required=True,
        help="Comma-separated method names, e.g. HAPPO_baseline,MAPPO_baseline,PRL_without_Norm",
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["early", "middle", "final"],
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional manual checkpoint override, e.g. backup_ep2500.pt",
    )

    parser.add_argument("--rollout_steps", type=int, default=20000)
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--probe_type", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save_datasets", action="store_true")

    parser.add_argument(
        "--out_root",
        type=str,
        default="probe_results",
        help="Root output dir.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    methods = parse_methods(args.methods)

    out_dir = Path(args.out_root) / f"{args.task}_seed{args.seed}_{args.stage}"
    manifest_path = out_dir / "manifest.json"

    build_manifest(
        task=args.task,
        seed=args.seed,
        methods=methods,
        stage=args.stage,
        checkpoint_override=args.checkpoint,
        manifest_path=manifest_path,
    )

    cmd = [
        "python",
        "harl/tools/probe_actor_embedding_uncertainty.py",
        "--manifest",
        str(manifest_path),
        "--out_dir",
        str(out_dir),
        "--rollout_steps",
        str(args.rollout_steps),
        "--train_epochs",
        str(args.train_epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--probe_types",
        args.probe_type,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]

    if args.deterministic:
        cmd.append("--deterministic")

    if args.save_datasets:
        cmd.append("--save_datasets")

    print("\nRunning probe command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    make_plots(
        out_dir=out_dir,
        stage=args.stage,
        probe_type=args.probe_type,
    )

    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()