# Humanoid Decomposition and Asymmetric Alignment

## Research question

Under a fixed Humanoid-v2 physical system, does asymmetric critic-to-actor
representation alignment become more useful as the same 17-dimensional action
space is split among more independently optimized actors?

The confirmatory trend comparison uses six nested decompositions and five model
interactions. `no_stop` is an optional diagnostic and is not part of the primary
five-way comparison.

## Invariants

Every run uses the same:

- MuJoCo `Humanoid-v2` dynamics, initial-state distribution, reward, termination,
  376-dimensional environment state, and 17 physical actuators;
- normalized global state as the complete input to every actor and the critic;
- canonical actuator ordering, reconstructed from each joint's `act_ids`;
- 10,000,000 environment steps, 200 rollout steps, 20 rollout environments,
  HAPPO objective, GAE, ValueNorm, optimizer settings, and evaluation protocol;
- seed within a matched-seed block.

No agent id, local observation, or dummy action coordinate is supplied in this
experiment. Different policy heads have their true action dimension.

## Nested decompositions

| Config | Agents | Anatomical action groups |
|---|---:|---|
| `1agent` | 1 | all 17 actuators |
| `3agents` | 3 | core; both legs; both arms |
| `5agents` | 5 | core; right leg; left leg; right arm; left arm |
| `7agents` | 7 | core; right hip; right knee; left hip; left knee; right arm; left arm |
| `11agents` | 11 | three core joints; right hip; right knee; left hip; left knee; right shoulder; right elbow; left shoulder; left elbow |
| `17x1` | 17 | one physical actuator per actor |

Each group at a finer level is a subset of exactly one group at the preceding
level. The group ordering is not used as MuJoCo action ordering: actions are
always placed into the canonical 0–16 actuator slots before `env.step`.

## Interaction treatments

Let `z_a(s)` be an actor encoder feature and `z_c(s)` the critic encoder feature
for the same current normalized global state. `LN` below is parameter-free layer
normalization over the feature dimension.

The alignment loss is:

`mean_agents MSE(LN(z_a(s)), LN(z_c(s)))`.

| Mode | Update semantics |
|---|---|
| `separate` | Independent actor and critic encoders; no alignment loss |
| `hard_share` | One encoder module shared by all actor heads and the value head |
| `critic_to_actor` | Add alignment to each PPO actor update; `z_c` is stop-gradient |
| `actor_to_critic` | Add mean actor alignment to the value update; every `z_a` is stop-gradient |
| `bidirectional` | Apply both one-way losses in their respective updates; each teacher side is stop-gradient |
| `no_stop` | After ordinary actor/value updates, perform an extra joint symmetric alignment step with gradients entering both sides |

There is no prediction/projector head, so all soft modes have identical model
parameter counts. Both soft coefficients default to 1.0 and must be varied only
in a separately declared sensitivity study.

## Capacity controls

Two complete grids are run:

1. `standard`: every independent encoder is a 3-layer 128-unit MLP. This measures
   the behavior of a conventional per-network architecture; total system
   parameters may grow with agent count.
2. `matched`: all actor encoders and the critic use a shared resolved width at
   depth 3, chosen to minimize absolute error from 1,000,000 unique trainable
   parameters. Shared tensors are counted once. The resolved width and exact
   model count are written into each run's saved `config.json`.

The parameter-matched condition controls total system capacity across agent
counts and across hard versus non-hard sharing. The standard condition remains
necessary because very narrow per-agent models at fine decompositions may be a
different practical regime even when total capacity is equal.

## Planned runs and endpoints

- Confirmatory modes: `separate`, `hard_share`, `critic_to_actor`,
  `actor_to_critic`, and `bidirectional`.
- Decompositions: 1, 3, 5, 7, 11, and 17 actors.
- Capacity modes: `standard` and `matched`.
- Matched seeds: 1, 2, 3, 4, and 5.
- Optional diagnostic: repeat the grid with `no_stop` only after the primary grid
  is complete.

Primary endpoint: mean evaluation episode return at the final scheduled
evaluation (40 deterministic episodes per seed). Secondary endpoints: area under
the evaluation-return curve through 10M steps, time-to-threshold using a threshold
declared after the 1-agent control but before inspecting treatment runs, and
wall-clock/parameter count diagnostics. Report individual seeds, mean, 95%
bootstrap confidence intervals over seeds, and paired seed-level treatment
differences within each decomposition/capacity condition.

No run is excluded for poor return. Infrastructure-failed runs may be rerun only
with the same commit and saved configuration; the failed run directory and reason
remain in the experiment ledger.

## Canonical config and launchers

- Config: `tuned_configs/mamujoco/Humanoid-v2-decomposition-alignment/config.json`
- Single run: `scripts/run_humanoid_decomposition_alignment.sh`
- Matrix generator: `scripts/generate_humanoid_alignment_jobs.py`
- Server procedure: `research/experiments/humanoid_decomposition_server.md`
