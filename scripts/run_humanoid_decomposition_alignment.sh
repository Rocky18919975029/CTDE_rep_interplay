#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "usage: $0 SEED DECOMPOSITION ALIGNMENT_MODE CAPACITY_MODE [GPU]" >&2
    echo "decomposition: 1agent|3agents|5agents|7agents|11agents|13agents|15agents|17x1" >&2
    echo "alignment: separate|hard_share|critic_to_actor|actor_to_critic|bidirectional|no_stop" >&2
    echo "capacity: standard|matched" >&2
    exit 2
fi

seed="$1"
decomposition="$2"
alignment_mode="$3"
capacity_mode="$4"
gpu="${5:-0}"

case "$decomposition" in
    1agent|3agents|5agents|7agents|11agents|13agents|15agents|17x1) ;;
    *) echo "invalid decomposition: $decomposition" >&2; exit 2 ;;
esac
case "$alignment_mode" in
    separate|hard_share|critic_to_actor|actor_to_critic|bidirectional|no_stop) ;;
    *) echo "invalid alignment mode: $alignment_mode" >&2; exit 2 ;;
esac
case "$capacity_mode" in
    standard|matched) ;;
    *) echo "invalid capacity mode: $capacity_mode" >&2; exit 2 ;;
esac

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="$repo_dir/tuned_configs/mamujoco/Humanoid-v2-decomposition-alignment/config.json"
python_bin="${PYTHON_BIN:-python}"
experiment_name="humanoid_${decomposition}_${alignment_mode}_${capacity_mode}"
extra_args=()
if [[ -n "${NUM_ENV_STEPS:-}" ]]; then
    extra_args+=(--num_env_steps "$NUM_ENV_STEPS")
fi
if [[ -n "${USE_EVAL:-}" ]]; then
    case "$USE_EVAL" in
        true|True|TRUE) eval_override="True" ;;
        false|False|FALSE) eval_override="False" ;;
        *) echo "USE_EVAL must be true or false" >&2; exit 2 ;;
    esac
    extra_args+=(--use_eval "$eval_override")
fi

cd "$repo_dir"
exec env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}" \
    "$python_bin" examples/train.py \
    --load_config "$config_path" \
    --exp_name "$experiment_name" \
    --seed "$seed" \
    --agent_conf "$decomposition" \
    --alignment_mode "$alignment_mode" \
    --capacity_mode "$capacity_mode" \
    "${extra_args[@]}"
