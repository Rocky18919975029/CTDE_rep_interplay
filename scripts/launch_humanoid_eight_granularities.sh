#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 SEED ALIGNMENT_MODE CAPACITY_MODE" >&2
    echo "example: $0 1 critic_to_actor standard" >&2
    exit 2
fi

seed="$1"
alignment_mode="$2"
capacity_mode="$3"

# Pair coarse and fine decompositions so every GPU carries 18 actor networks.
decompositions=(
    1agent 17x1
    3agents 15agents
    5agents 13agents
    7agents 11agents
)
gpus=(0 0 1 1 2 2 3 3)

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner="$repo_dir/scripts/run_humanoid_decomposition_alignment.sh"
timestamp="$(date +%Y-%m-%d-%H-%M-%S)"
log_dir="$repo_dir/launch_logs/${timestamp}_seed-${seed}_${alignment_mode}_${capacity_mode}"

if [[ "${DRY_RUN:-false}" == "true" ]]; then
    for job in 0 1 2 3 4 5 6 7; do
        printf '%q ' "$runner" "$seed" "${decompositions[$job]}" \
            "$alignment_mode" "$capacity_mode" "${gpus[$job]}"
        printf '\n'
    done
    exit 0
fi

mkdir -p "$log_dir"
pids=()

terminate_children() {
    if [[ ${#pids[@]} -gt 0 ]]; then
        kill "${pids[@]}" 2>/dev/null || true
    fi
}
trap terminate_children INT TERM

for job in 0 1 2 3 4 5 6 7; do
    decomposition="${decompositions[$job]}"
    gpu="${gpus[$job]}"
    log_file="$log_dir/gpu${gpu}_${decomposition}.log"
    "$runner" "$seed" "$decomposition" "$alignment_mode" \
        "$capacity_mode" "$gpu" >"$log_file" 2>&1 &
    pids+=("$!")
    echo "GPU $gpu: $decomposition, PID ${pids[$job]}, log $log_file"
done

failed=0
for job in 0 1 2 3 4 5 6 7; do
    if wait "${pids[$job]}"; then
        echo "GPU ${gpus[$job]} (${decompositions[$job]}) completed"
    else
        status="$?"
        echo "GPU ${gpus[$job]} (${decompositions[$job]}) failed with status $status" >&2
        failed=1
    fi
done

exit "$failed"
