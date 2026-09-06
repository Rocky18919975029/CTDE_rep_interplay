#!/usr/bin/env python3
"""Print the preregistered experiment matrix as one shell command per run."""

import argparse
import shlex


DECOMPOSITIONS = ["1agent", "3agents", "5agents", "7agents", "11agents", "17x1"]
ALIGNMENT_MODES = [
    "separate",
    "hard_share",
    "critic_to_actor",
    "actor_to_critic",
    "bidirectional",
]
CAPACITY_MODES = ["standard", "matched"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--include-no-stop", action="store_true")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    modes = ALIGNMENT_MODES + (["no_stop"] if args.include_no_stop else [])
    for seed in args.seeds:
        for decomposition in DECOMPOSITIONS:
            for alignment_mode in modes:
                for capacity_mode in CAPACITY_MODES:
                    command = [
                        "./scripts/run_humanoid_decomposition_alignment.sh",
                        str(seed),
                        decomposition,
                        alignment_mode,
                        capacity_mode,
                        args.gpu,
                    ]
                    print(" ".join(shlex.quote(part) for part in command))


if __name__ == "__main__":
    main()
