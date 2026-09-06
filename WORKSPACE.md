# PRL New Paper Workspace

This repository is the clean starting point for a new paper built from the local
PRL research implementation.

## Source provenance

- Snapshot source: `/Users/zeshenghong/Downloads/PRL`
- Snapshot date: 2026-09-06
- Imported files: 1,087
- Aggregate source digest (SHA-256 over the ordered per-file digests):
  `99df012c96e16a30055402472d2545ce423e2ef23dea2d6d525d76538a002913`
- The source directory contained no Git metadata, dependency lock file, or
  packaging metadata.
- Machine-generated caches (`.DS_Store`, `__pycache__`, and `*.pyc`) and prior
  result directories were deliberately excluded.

The imported research code was not behaviorally modified while constructing the
workspace. Packaging metadata, ignore rules, and the research scaffold are new.

## Baseline validation

- All 201 Python files under `harl/` and `examples/` compile successfully with
  macOS `/usr/bin/python3` 3.9.6.
- The training environments have not yet been executed in this workspace because
  the original Python/Conda dependency specification was not present.
- The machine's first `python3` on `PATH` points to an obsolete Python 3.4 binary;
  create and activate an explicit environment before running experiments.

## Research layout

- `harl/`, `examples/`, `tuned_configs/`: imported implementation and configs.
- `research/experiments/`: immutable experiment specifications and launch notes.
- `research/analysis/`: analysis code and figure/table provenance.
- `paper/`: manuscript source, figures, and submission notes.
- `results/`: generated locally and ignored by Git.

## Reproducibility contract

Every reported experiment should record:

1. the Git commit and exact configuration file;
2. the environment name and dependency lock or exported environment;
3. the complete seed set and training/evaluation budget;
4. raw event logs plus a compact machine-readable metric export;
5. checkpoint policy and hardware/runtime information.

Comparisons must use matched seeds and identical non-treatment hyperparameters.
Any intentional capacity, loss, input, or target change belongs in the experiment
specification before a run starts.

