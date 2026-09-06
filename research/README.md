# Research Index

This directory is the authoritative index for the new paper's research record.

Before launching the first large experiment, define the central hypothesis,
primary comparison, benchmark suite, evaluation metric, and stopping budget here.
Do not treat exploratory runs as confirmatory evidence unless they were specified
in advance.

## Workflow

1. Write a concise experiment specification in `experiments/`.
2. Copy or derive a committed JSON config for every treatment and control.
3. Run matched seeds from the same Git commit and environment.
4. Put analysis scripts and metric definitions in `analysis/`.
5. Link each paper figure and table back to its script and raw run directories.

Legacy PRL results remain under `/Users/zeshenghong/Downloads/results`; they were
not copied into this repository and should be treated as prior exploratory data
unless their code/environment provenance can be reconstructed.

