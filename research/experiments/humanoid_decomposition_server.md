# Server Upload and Run Procedure

## Upload

The repository currently has no Git remote. Either push this branch to a new
remote and clone it on the server, or copy the workspace directly. For a direct
copy, replace `USER`, `SERVER`, and the destination path:

```bash
rsync -azh --info=progress2 \
  --exclude '.git/' \
  --exclude 'results/' \
  /Users/zeshenghong/Documents/Codex/PRL_New_Paper/ \
  USER@SERVER:/path/to/PRL_New_Paper/
```

Do not use `--delete` against an existing server result directory.

## Environment

Activate the server environment that already runs the original PRL Humanoid
experiments. The imported repository did not include a dependency lock, so that
known-good environment is safer than silently constructing a new one.

```bash
ssh USER@SERVER
cd /path/to/PRL_New_Paper
conda activate YOUR_EXISTING_PRL_ENV
python -m pip install -e .
python -c "import torch, gym, numpy; print(torch.__version__, gym.__version__, numpy.__version__)"
```

The launch script and `examples/train.py` explicitly put this checkout first on
`PYTHONPATH`. This is important when the Conda environment also contains an
editable install of an older HARL checkout. Every run prints `HARL source:`; it
must point inside `/path/to/PRL_New_Paper`, never an older experiment directory.

If your server uses `mujoco_py`, keep the same MuJoCo library and license setup as
the prior PRL runs. The experiment intentionally targets the existing
`Humanoid-v2` API rather than migrating to a newer Gym task.

## Tests and one-update smoke run

```bash
python -m unittest discover -s tests -v

NUM_ENV_STEPS=4000 USE_EVAL=false \
  ./scripts/run_humanoid_decomposition_alignment.sh \
  1 3agents critic_to_actor standard 0
```

With 20 rollout environments and rollout length 200, 4,000 environment steps is
one training update. Check that the printed action dimensions sum to 17, the
printed actor/critic observation shapes are identical, the parameter accounting
check passes, and the run exits without a tensor-shape error.

Before the large grid, also smoke-test the two structurally special modes:

```bash
NUM_ENV_STEPS=4000 USE_EVAL=false \
  ./scripts/run_humanoid_decomposition_alignment.sh \
  1 11agents hard_share matched 0

NUM_ENV_STEPS=4000 USE_EVAL=false \
  ./scripts/run_humanoid_decomposition_alignment.sh \
  1 17x1 actor_to_critic standard 0
```

## Full runs

Single run, with the final argument selecting the visible GPU:

```bash
./scripts/run_humanoid_decomposition_alignment.sh \
  1 7agents critic_to_actor matched 0
```

Generate the 300-command primary grid (6 decompositions × 5 modes × 2 capacity
conditions × 5 seeds):

```bash
python scripts/generate_humanoid_alignment_jobs.py \
  --seeds 1 2 3 4 5 > primary_jobs.sh
```

The generated file is a manifest, not a scheduler: submit one line per GPU job
using the server's existing Slurm/Tmux workflow. Generate the optional no-stop
diagnostic only when needed:

```bash
python scripts/generate_humanoid_alignment_jobs.py \
  --seeds 1 2 3 4 5 --include-no-stop > all_jobs_with_no_stop.sh
```

Results are rooted at `results/humanoid_decomposition_alignment/mamujoco/`.
Every run name includes decomposition, alignment mode, and capacity mode; HARL's
result path additionally records seed and timestamp. Preserve the generated
`config.json`, TensorBoard logs, `training_time.txt`, and checkpoints together.
