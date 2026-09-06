import unittest
import sys
import types
import importlib.util
from pathlib import Path

try:
    import numpy as np
except ImportError:  # The local macOS system Python is intentionally dependency-free.
    np = None
    sys.modules["numpy"] = types.SimpleNamespace()

module_spec = importlib.util.spec_from_file_location(
    "mamujoco_obsk",
    Path(__file__).parents[1]
    / "harl"
    / "envs"
    / "mamujoco"
    / "multiagent_mujoco"
    / "obsk.py",
)
obsk = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(obsk)
build_actions = obsk.build_actions
get_parts_and_edges = obsk.get_parts_and_edges


DECOMPOSITIONS = ["1agent", "3agents", "5agents", "7agents", "11agents", "17x1"]


def actuator_sets(partition):
    return [{node.act_ids for node in agent} for agent in partition]


class HumanoidDecompositionTest(unittest.TestCase):
    def test_agent_counts_and_complete_action_coverage(self):
        for label, expected_count in zip(DECOMPOSITIONS, [1, 3, 5, 7, 11, 17]):
            partition, _, _ = get_parts_and_edges("Humanoid-v2", label)
            self.assertEqual(len(partition), expected_count)
            actuator_ids = [
                node.act_ids for agent in partition for node in agent
            ]
            self.assertEqual(sorted(actuator_ids), list(range(17)))

    def test_each_level_only_splits_prior_groups(self):
        partitions = [
            get_parts_and_edges("Humanoid-v2", label)[0]
            for label in DECOMPOSITIONS
        ]
        for coarse, fine in zip(partitions, partitions[1:]):
            coarse_sets = actuator_sets(coarse)
            for fine_group in actuator_sets(fine):
                self.assertEqual(
                    sum(fine_group <= coarse_group for coarse_group in coarse_sets),
                    1,
                )

    def test_action_builder_restores_canonical_order(self):
        if np is None:
            self.skipTest("NumPy is required for the action-array check")
        for label in DECOMPOSITIONS:
            partition, _, _ = get_parts_and_edges("Humanoid-v2", label)
            actions = [
                np.asarray([node.act_ids + 0.25 for node in agent], dtype=np.float32)
                for agent in partition
            ]
            flat = build_actions(partition, actions)
            np.testing.assert_allclose(flat, np.arange(17) + 0.25)


if __name__ == "__main__":
    unittest.main()
