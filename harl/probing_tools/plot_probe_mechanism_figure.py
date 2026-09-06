# harl/tools/plot_probe_mechanism_figure.py

from __future__ import annotations

import argparse
import re
import math
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


TARGET_DISPLAY = {
    "partial_observation_state": "Current State Reconstruction",
    "teammate_action": "Teammate-Action Reconstruction",
    "transition_next_critic_latent": "Pred. Next Critic Embedding",
}

TARGET_ORDER = [
    "partial_observation_state",
    "teammate_action",
    "transition_next_critic_latent",
]

METHOD_ORDER = [
    "HAPPO",
    "HAPPO_deep",
    "MAPPO",
    "PRL",
]

METHOD_COLOR = {
    "HAPPO": "#D7E3F3",       
    "HAPPO_deep": "#A9C4E2",  
    "MAPPO": "#6F93C5",       
    "PRL": "#2F5597",         
}

TASK_DISPLAY = {
    "grf_counterattack_hard": "GRF: Counterattack Hard",
    "academy_counterattack_hard": "GRF: Counterattack Hard",
    "grf_3_vs_1_with_keeper": "GRF: 3 vs 1 w/ Keeper",
    "academy_3_vs_1_with_keeper": "GRF: 3 vs 1 w/ Keeper",
    "dexhands_SHO": "DexHands: ShadowHandOver",
    "ShadowHandOver": "DexHands: ShadowHandOver",
    "mamujoco_humanoid_17x1": "MA-MuJoCo: Humanoid-17x1",
    "humanoid17x1": "MA-MuJoCo: Humanoid-17x1",
}



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="probe_results",
        help="Root directory containing <task>_seed<k>_final folders.",
    )

    parser.add_argument(
        "--probe_type",
        type=str,
        default="linear",
        choices=["linear", "mlp"],
        help="Which probe type to plot.",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="r2",
        choices=["r2", "norm_mse"],
        help="Metric to plot. Recommended: r2.",
    )

    parser.add_argument(
        "--layout",
        type=str,
        default="2x2",
        help="Panel layout, e.g. 2x2, 4x1, 1x4.",
    )

    parser.add_argument(
        "--task_order",
        nargs="*",
        default=None,
        help="Optional manual task order.",
    )

    parser.add_argument(
        "--save_name",
        type=str,
        default=None,
        help="Output file name without suffix.",
    )

    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title.",
    )

    return parser.parse_args()



def parse_layout(layout: str):
    """
    Parse layout strings such as:
      2x2, 4x1, 1x4

    Return:
      n_rows, n_cols
    """
    layout = layout.lower().strip()

    if "x" not in layout:
        raise ValueError(
            f"Invalid layout: {layout}. Expected format like 2x2 or 4x1."
        )

    parts = layout.split("x")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid layout: {layout}. Expected format like 2x2 or 4x1."
        )

    try:
        n_rows = int(parts[0])
        n_cols = int(parts[1])
    except ValueError:
        raise ValueError(
            f"Invalid layout: {layout}. Rows and columns must be integers."
        )

    if n_rows <= 0 or n_cols <= 0:
        raise ValueError(
            f"Invalid layout: {layout}. Rows and columns must be positive."
        )

    return n_rows, n_cols


def get_fig_size(n_rows: int, n_cols: int):
    """
    Publication-oriented figure size.

    For the default 2x2 layout, use a wider and slightly shorter canvas
    so each panel has more horizontal room while keeping the figure compact
    for a NeurIPS page.
    """
    if n_rows == 2 and n_cols == 2:
        return 13.6, 7.2

    if n_rows == 1 and n_cols == 4:
        return 15.5, 3.8

    if n_rows == 4 and n_cols == 1:
        return 7.2, 12.8

    fig_w = 6.8 * n_cols
    fig_h = 3.6 * n_rows
    return fig_w, fig_h




def parse_task_seed_from_parent(parent_name: str):
    """
    Parse folder names like:
      dexhands_SHO_seed1_final
      grf_counterattack_hard_seed2_final
      humanoid17x1_seed3_final
    """
    m = re.match(r"(.+)_seed(\d+)_final$", parent_name)
    if m is None:
        raise ValueError(f"Cannot parse task/seed from folder name: {parent_name}")

    task = m.group(1)
    seed = int(m.group(2))
    return task, seed


def discover_task_dirs(root: Path) -> List[Path]:
    task_dirs = sorted([p for p in root.glob("*_seed*_final") if p.is_dir()])

    if len(task_dirs) == 0:
        raise FileNotFoundError(f"No *_seed*_final folders found under: {root}")

    return task_dirs


def discover_method_csvs(root: Path) -> List[Path]:
    """
    Discover per-method csv files:
      <root>/<task>_seed<k>_final/probe_results_<method>_final.csv

    Skip:
      probe_results_all.csv
      *_summary.csv
    """
    csvs = []

    for task_dir in discover_task_dirs(root):
        for csv_path in sorted(task_dir.glob("probe_results_*.csv")):
            name = csv_path.name

            if name == "probe_results_all.csv":
                continue

            if name.endswith("_summary.csv"):
                continue

            csvs.append(csv_path)

    return csvs


def discover_all_csvs(root: Path) -> List[Path]:
    """
    Fallback: use probe_results_all.csv if per-method csvs are not found.
    """
    csvs = sorted(root.glob("*_seed*_final/probe_results_all.csv"))
    return csvs


def normalize_method_name(name: str) -> str:
    """
    Normalize common method names produced by different scripts.
    """
    raw = str(name)

    mapping = {
        "HAPPO_baseline": "HAPPO",
        "HAPPO": "HAPPO",

        "HAPPO_deep_baseline": "HAPPO_deep",
        "HAPPO_deep": "HAPPO_deep",
        "HAPPO_deep_final": "HAPPO_deep",

        "MAPPO_baseline": "MAPPO",
        "MAPPO": "MAPPO",

        "PRL_ours": "PRL",
        "PRL": "PRL",
        "PRL_without_Norm": "PRL",
        "PRL_lembda_0.5": "PRL",
    }

    return mapping.get(raw, raw)



def load_all_results(root: Path, probe_type: str) -> pd.DataFrame:
    """
    Load all probe csv files.

    Priority:
      1. per-method csvs
      2. probe_results_all.csv fallback
    """
    method_csvs = discover_method_csvs(root)

    if len(method_csvs) > 0:
        csvs = method_csvs
        print(f"\n[Info] Loading per-method CSV files: {len(csvs)}")
    else:
        csvs = discover_all_csvs(root)
        print(f"\n[Info] Loading probe_results_all.csv files: {len(csvs)}")

    if len(csvs) == 0:
        raise FileNotFoundError(f"No probe csv files found under: {root}")

    rows = []

    for csv_path in csvs:
        task, seed = parse_task_seed_from_parent(csv_path.parent.name)

        df = pd.read_csv(csv_path)

        if len(df) == 0:
            continue

        if "probe_type" in df.columns:
            df = df[df["probe_type"] == probe_type].copy()

        if len(df) == 0:
            continue

        if "method" not in df.columns:
            raise ValueError(f"CSV missing `method` column: {csv_path}")

        df["method"] = df["method"].map(normalize_method_name)
        df["task"] = task
        df["seed"] = seed
        df["source_csv"] = str(csv_path)

        rows.append(df)

    if len(rows) == 0:
        raise RuntimeError("No valid probe rows found after filtering.")

    out = pd.concat(rows, ignore_index=True)

    print("\n[Info] Loaded rows by task/method:")
    print(
        out.groupby(["task", "method"])
        .size()
        .reset_index(name="n_rows")
        .sort_values(["task", "method"])
        .to_string(index=False)
    )

    print("\n[Info] Loaded rows by task/method/target:")
    print(
        out.groupby(["task", "method", "target_name"])
        .size()
        .reset_index(name="n_rows")
        .sort_values(["task", "method", "target_name"])
        .to_string(index=False)
    )

    return out




def aggregate_probe_results(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Step 1: average over agents within each task/seed/method/target.
    Step 2: aggregate over seeds.
    """
    required_cols = {"task", "seed", "method", "target_name", metric}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    by_seed = (
        df.groupby(["task", "seed", "method", "target_name"], as_index=False)
        .agg(metric_value=(metric, "mean"))
    )

    agg = (
        by_seed.groupby(["task", "method", "target_name"], as_index=False)
        .agg(
            mean=("metric_value", "mean"),
            std=("metric_value", "std"),
            n_seeds=("metric_value", "count"),
        )
    )

    agg["std"] = agg["std"].fillna(0.0)
    agg["target_display"] = agg["target_name"].map(TARGET_DISPLAY).fillna(agg["target_name"])
    agg["task_display"] = agg["task"].map(TASK_DISPLAY).fillna(agg["task"])

    print("\n[Info] Aggregated results:")
    print(
        agg.sort_values(["task", "target_name", "method"])
        .to_string(index=False)
    )

    return agg




def get_task_order(agg: pd.DataFrame, manual_task_order: Optional[List[str]] = None):
    discovered = list(agg["task"].drop_duplicates())

    if manual_task_order is not None and len(manual_task_order) > 0:
        order = [t for t in manual_task_order if t in discovered]
        order += [t for t in discovered if t not in order]
        return order

    preferred = [
        "grf_counterattack_hard",
        "academy_counterattack_hard",
        "academy_3_vs_1_with_keeper",
        "grf_3_vs_1_with_keeper",
        "dexhands_SHO",
        "ShadowHandOver",
        "mamujoco_humanoid_17x1",
        "humanoid17x1",
    ]

    order = [t for t in preferred if t in discovered]
    order += [t for t in discovered if t not in order]

    return order


def get_method_order(agg: pd.DataFrame):
    discovered = list(agg["method"].drop_duplicates())

    order = [m for m in METHOD_ORDER if m in discovered]
    order += [m for m in discovered if m not in order]

    return order




def report_missing_combinations(agg: pd.DataFrame, tasks: List[str], methods: List[str]):
    missing = []

    for task in tasks:
        for method in methods:
            for tgt in TARGET_ORDER:
                row = agg[
                    (agg["task"] == task)
                    & (agg["method"] == method)
                    & (agg["target_name"] == tgt)
                ]

                if len(row) == 0:
                    missing.append((task, method, tgt))

    if len(missing) > 0:
        print("\n[Warning] Missing task-method-target combinations:")
        for task, method, tgt in missing:
            print(f"  task={task}, method={method}, target={tgt}")



def plot_mechanism_figure(
    agg: pd.DataFrame,
    metric: str,
    save_pdf: Path,
    save_png: Path,
    task_order: Optional[List[str]] = None,
    title: Optional[str] = None,
    layout: str = "2x2",
):
    tasks = get_task_order(agg, task_order)
    methods = [m for m in get_method_order(agg) if m != "HAPPO_deep"]

    report_missing_combinations(agg, tasks, methods)

    n_tasks = len(tasks)
    n_rows, n_cols = parse_layout(layout)

    if n_tasks > n_rows * n_cols:
        raise ValueError(
            f"Layout {layout} has only {n_rows * n_cols} panels, "
            f"but {n_tasks} tasks need to be plotted."
        )

    fig_w, fig_h = get_fig_size(n_rows, n_cols)

    plt.rcParams.update({
        "font.size": 11.5,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        sharey=True,
    )

    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    max_x = float((agg["mean"] + agg["std"]).max())
    min_x = float((agg["mean"] - agg["std"]).min())

    if metric == "r2":
        x_left = min(0.0, min_x - 0.04)
        x_right = max(1.0, max_x + 0.04)
    else:
        x_left = min(0.0, min_x - 0.04)
        x_right = max(max_x + 0.04, 0.2)

    factor_gap = 0.5   # 可再试 0.75 / 0.72
    base_y = [k * factor_gap for k in range(len(TARGET_ORDER))]
    n_methods = len(methods)

    total_band = 0.32
    bar_h = total_band / max(n_methods, 1)

    for ax, task in zip(axes, tasks):
        sub = agg[agg["task"] == task].copy()

        for i, method in enumerate(methods):
            color = METHOD_COLOR.get(method, "#999999")

            for j, tgt in enumerate(TARGET_ORDER):
                row = sub[
                    (sub["method"] == method)
                    & (sub["target_name"] == tgt)
                ]

                if len(row) == 0:
                    continue

                val = float(row["mean"].iloc[0])
                err = float(row["std"].iloc[0])

                ypos = base_y[j] + (i - (n_methods - 1) / 2) * bar_h

                ax.barh(
                    ypos,
                    val,
                    height=bar_h * 0.88,
                    xerr=err,
                    color=color,
                    edgecolor="black",
                    linewidth=0.45,
                    capsize=2.0,
                    error_kw={
                        "elinewidth": 0.8,
                        "capthick": 0.8,
                    },
                    alpha=0.98,
                )

        ax.set_title(
            TASK_DISPLAY.get(task, task),
            fontsize=12,
            fontweight="semibold",
            pad=4,
        )

        ax.set_yticks(base_y)
        ax.set_yticklabels(
            [TARGET_DISPLAY.get(t, t) for t in TARGET_ORDER],
            fontsize=11,
        )
        ax.invert_yaxis()
        ax.margins(y=0.06)

        ax.set_xlim(x_left, x_right)
        ax.grid(axis="x", linestyle="--", alpha=0.25, linewidth=0.7)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["bottom"].set_linewidth(0.7)

        ax.tick_params(axis="x", labelsize=10.5, width=0.7, length=3)
        ax.tick_params(axis="y", labelsize=11, width=0.7, length=0)

    for ax in axes[n_tasks:]:
        ax.axis("off")


    for idx, ax in enumerate(axes[:n_tasks]):
        col_idx = idx % n_cols
        if col_idx != 0:
            ax.tick_params(axis="y", labelleft=False)

    for row_idx in range(n_rows):
        ax_idx = row_idx * n_cols
        if ax_idx < n_tasks:
            axes[ax_idx].set_ylabel("Probed target", fontsize=12)

    handles = [
        Patch(
            facecolor=METHOD_COLOR.get(m, "#999999"),
            edgecolor="black",
            linewidth=0.45,
            label=m.replace("_deep", "-Deep"),
        )
        for m in methods
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(methods),
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=1.4,
        handlelength=1.2,
        handletextpad=0.35,
        borderaxespad=0.0,
        fontsize=11,
    )

    if title is not None:
        fig.suptitle(
            title,
            fontsize=13,
            fontweight="semibold",
            y=0.988,
        )
        plt.tight_layout(rect=[0.02, 0.025, 0.995, 0.94], h_pad=0.85, w_pad=0.85)
    else:
        plt.tight_layout(rect=[0.02, 0.025, 0.995, 0.98], h_pad=0.85, w_pad=0.85)

    save_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_pdf, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.savefig(save_png, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close()

    print(f"\n[Saved] PDF: {save_pdf}")
    print(f"[Saved] PNG: {save_png}")


def main():
    args = parse_args()

    root = Path(args.root)

    df = load_all_results(
        root=root,
        probe_type=args.probe_type,
    )

    agg = aggregate_probe_results(
        df=df,
        metric=args.metric,
    )

    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_name is None:
        save_name = f"mechanism_probe_{args.probe_type}_{args.metric}"
    else:
        save_name = args.save_name

    save_pdf = out_dir / f"{save_name}.pdf"
    save_png = out_dir / f"{save_name}.png"
    save_csv = out_dir / f"{save_name}_aggregated.csv"

    agg.to_csv(save_csv, index=False)
    print(f"\n[Saved] Aggregated CSV: {save_csv}")

    plot_mechanism_figure(
        agg=agg,
        metric=args.metric,
        save_pdf=save_pdf,
        save_png=save_png,
        task_order=args.task_order,
        title=args.title,
        layout=args.layout,
    )


if __name__ == "__main__":
    main()