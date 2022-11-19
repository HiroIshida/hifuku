from math import floor

import numpy as np

from hifuku.margin import CoverageResult


def test_coverage_result():

    for error in np.linspace(-0.1, 0.1, 200):

        def func_gt(x: np.ndarray) -> np.ndarray:
            return x - 0.5

        def func_est(x: np.ndarray) -> np.ndarray:
            return (x - (0.5 + error)) + np.random.randn(len(x)) * 0.03

        n_sample = 100
        xs = np.random.rand(n_sample)
        values_gt = func_gt(xs)
        values_est = func_est(xs)

        result = CoverageResult(values_gt, values_est, 0.0)
        assert sum(result.false_postive_bools) + sum(result.false_negative_bools) + sum(
            result.true_positive_bools
        ) + sum(result.true_negative_bools) == len(result)
        rate_threshold = 0.05
        margin = result.determine_margin(rate_threshold)

        result = CoverageResult(values_gt, values_est + margin, 0.0)
        n_fp = sum(result.false_postive_bools)
        n_fp_expected = min(floor(n_sample * rate_threshold), np.sum(result.false_postive_bools))
        assert n_fp == n_fp_expected


if __name__ == "__main__":
    test_coverage_result()
