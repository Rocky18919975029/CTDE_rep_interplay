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

Generate the 400-command primary grid (8 decompositions × 5 modes × 2 capacity
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

## Four-GPU, eight-granularity same-seed launch

Run all eight decompositions concurrently, with two processes on each GPU. Coarse
and fine decompositions are paired so each GPU carries 18 actor networks:

```bash
./scripts/launch_humanoid_eight_granularities.sh \
  1 critic_to_actor standard
```

The mapping is GPU 0 = 1+17, GPU 1 = 3+15, GPU 2 = 5+13, and GPU 3 = 7+11.

Run it inside `tmux` so an SSH disconnect does not terminate the jobs:

```bash
tmux new-session -d -s humanoid-s1-c2a \
  './scripts/launch_humanoid_eight_granularities.sh 1 critic_to_actor standard'
```

Per-process stdout/stderr is written under `launch_logs/`.

## Live eight-panel reward monitor

The live monitor overlays two methods in every decomposition panel by default:
orange is the HAPPO baseline (`separate`) and blue is critic-to-actor alignment.
It selects the newest matching run independently for every decomposition and
method. Consequently, completed 1-agent and 3-agent runs remain visible while
new 5--17-agent reruns update live. It reads HARL's flushed `progress.txt` and
compares evaluation episode return at the same environment-step coordinates.
Every panel uses the common 0--10 million-step x-axis, so a partial live curve
cannot look artificially as long as a completed curve. Optional `auto` mode reads
TensorBoard's training-return event as a fallback before the first evaluation
interval.

Install the two plotting dependencies in the experiment environment if needed,
then start the localhost-only web view in a second `tmux` session:

```bash
python -m pip install matplotlib tensorboard

tmux new-session -d -s humanoid-rewards \
  'cd ~/CTDE_rep_interplay && conda run -n marl_trpo \
   python scripts/live_plot_humanoid_rewards.py \
   --seed 1 --capacity-mode standard \
   --refresh-seconds 10 --port 8765'
```

On the local computer, forward the monitor port and open
`http://127.0.0.1:8765`:

```bash
ssh -N -L 8765:127.0.0.1:8765 zeshenghong@ada4090g1
```

Evaluation return is the default so the two methods always remain comparable.
Use `--metric auto` to enable a temporary training-return fallback,
`--metric train` for training returns only, or `--x-axis hours` for elapsed
wall-clock time instead of environment steps. To create or continually update a
PNG file, pass `--output results/humanoid_rewards_live.png`; add `--once` to
write it once and exit. `--alignment-modes` can select another set of methods;
its default is `separate critic_to_actor`.
