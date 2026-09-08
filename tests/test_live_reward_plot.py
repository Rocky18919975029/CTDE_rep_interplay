"""Tests for the live Humanoid reward monitor's dependency-free helpers."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "live_plot_humanoid_rewards.py"
SPEC = importlib.util.spec_from_file_location("live_plot_humanoid_rewards", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LiveRewardPlotTest(unittest.TestCase):
    def test_find_latest_matching_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = (
                root
                / "mamujoco"
                / "Humanoid-v2-3agents"
                / "happo"
                / "humanoid_3agents_critic_to_actor_standard"
            )
            older = experiment / "seed-00001-2026-09-07-10-00-00"
            newer = experiment / "seed-00001-2026-09-07-11-00-00"
            wrong_seed = experiment / "seed-00002-2026-09-07-12-00-00"
            for path in (older, newer, wrong_seed):
                path.mkdir(parents=True)

            actual = MODULE.find_latest_run(
                root, "3agents", 1, "critic_to_actor", "standard"
            )

            self.assertEqual(actual, newer)

    def test_find_latest_run_ignores_other_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = (
                root
                / "humanoid_3agents_separate_standard"
                / "seed-00001-2026-09-07-11-00-00"
            )
            path.mkdir(parents=True)
            actual = MODULE.find_latest_run(
                root, "3agents", 1, "critic_to_actor", "standard"
            )
            self.assertIsNone(actual)

    def test_completed_and_live_decompositions_are_selected_independently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = (
                root
                / "Humanoid-v2-1agent"
                / "happo"
                / "humanoid_1agent_separate_standard"
                / "seed-00001-2026-09-07-10-00-00"
            )
            live = (
                root
                / "Humanoid-v2-5agents"
                / "happo"
                / "humanoid_5agents_separate_standard"
                / "seed-00001-2026-09-08-19-40-30"
            )
            completed.mkdir(parents=True)
            live.mkdir(parents=True)

            self.assertEqual(
                MODULE.find_latest_run(root, "1agent", 1, "separate", "standard"),
                completed,
            )
            self.assertEqual(
                MODULE.find_latest_run(root, "5agents", 1, "separate", "standard"),
                live,
            )

    def test_moving_average(self):
        self.assertEqual(MODULE.moving_average([1.0, 3.0, 8.0], 2), [1.0, 2.0, 5.5])

    def test_read_eval_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            progress = Path(temporary) / "progress.txt"
            progress.write_text("100000,12.5\n200000,25.75\n", encoding="utf-8")
            points, errors = MODULE.read_eval_progress(progress)
            self.assertEqual([point[0] for point in points], [100000, 200000])
            self.assertEqual([point[2] for point in points], [12.5, 25.75])
            self.assertEqual(errors, [])

    def test_matching_tag_accepts_tensorboardx_variants(self):
        target = "eval_average_episode_rewards"
        self.assertEqual(
            MODULE._matching_tag(["other", target], target),
            target,
        )
        self.assertEqual(
            MODULE._matching_tag(
                ["eval_average_episode_rewards/eval_average_episode_rewards"],
                target,
            ),
            "eval_average_episode_rewards/eval_average_episode_rewards",
        )

    def test_comparison_styles_are_distinct(self):
        baseline = MODULE.method_style("separate", 0)
        critic_to_actor = MODULE.method_style("critic_to_actor", 1)
        self.assertEqual(baseline["label"], "HAPPO baseline")
        self.assertNotEqual(baseline["color"], critic_to_actor["color"])


if __name__ == "__main__":
    unittest.main()
