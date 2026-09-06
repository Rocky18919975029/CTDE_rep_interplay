import unittest

from harl.utils.decomposition_experiment import (
    experiment_parameter_count,
    resolve_parameter_matched_width,
)


class ParameterMatchingTest(unittest.TestCase):
    def test_resolver_selects_nearest_integer_width(self):
        target = 1_000_000
        for action_dims in ([17], [3, 8, 6], [1] * 17):
            for hard_share in (False, True):
                width, count = resolve_parameter_matched_width(
                    376,
                    action_dims,
                    target,
                    3,
                    hard_share=hard_share,
                )
                candidates = [
                    experiment_parameter_count(
                        376,
                        action_dims,
                        candidate_width,
                        3,
                        hard_share=hard_share,
                    )
                    for candidate_width in range(max(8, width - 1), width + 2)
                ]
                self.assertEqual(abs(count - target), min(abs(x - target) for x in candidates))

    def test_soft_modes_share_the_same_count_formula(self):
        count = experiment_parameter_count(376, [3, 8, 6], 128, 3)
        self.assertEqual(count, experiment_parameter_count(376, [3, 8, 6], 128, 3))
        self.assertLess(
            experiment_parameter_count(376, [3, 8, 6], 128, 3, hard_share=True),
            count,
        )


if __name__ == "__main__":
    unittest.main()
