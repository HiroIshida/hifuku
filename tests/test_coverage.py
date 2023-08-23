from math import floor
from typing import List, Tuple

import numpy as np

from hifuku.coverage import CoverageResult, compute_coverage_and_fp_jit


def test_coverage_result():

    for rate_threshold in [1e-3, 0.5]:
        for error in np.linspace(-0.1, 0.1, 200):

            def func_gt(x: np.ndarray) -> np.ndarray:
                return x - 0.5

            def func_est(x: np.ndarray) -> np.ndarray:
                return (x - (0.5 + error)) + np.random.randn(len(x)) * 0.03

            n_sample = 1000
            xs = np.random.rand(n_sample)
            values_gt = func_gt(xs)
            values_est = func_est(xs)

            result = CoverageResult(values_gt, values_est, 0.0)

            assert sum(result.false_postive_bools) + sum(result.false_negative_bools) + sum(
                result.true_positive_bools
            ) + sum(result.true_negative_bools) == len(result)
            # rate_threshold = 0.05
            margin = result.determine_margin(rate_threshold)

            result = CoverageResult(values_gt, values_est + margin, 0.0)

            # sum must equal
            n_sum = (
                sum(result.true_positive_bools)
                + sum(result.true_negative_bools)
                + sum(result.false_postive_bools)
                + sum(result.false_negative_bools)
            )
            assert n_sum == n_sample

            n_fp = sum(result.false_postive_bools)

            total_positive_num = sum(result.true_positive_bools) + sum(result.false_postive_bools)
            n_fp_expected = min(
                floor(total_positive_num * rate_threshold), np.sum(result.false_postive_bools)
            )
            assert n_fp == n_fp_expected


def test_coverage_result_int_case():
    n_sample = 1000
    values_gt = np.random.randint(5, size=(n_sample,))
    values_est = np.random.randint(5, size=(n_sample,))
    result = CoverageResult(values_gt, values_est, 2)
    n_sum = (
        sum(result.true_positive_bools)
        + sum(result.true_negative_bools)
        + sum(result.false_postive_bools)
        + sum(result.false_negative_bools)
    )
    assert n_sum == n_sample


def compute_coverage_and_fp_naive(
    margins: np.ndarray, coverage_results: List[CoverageResult], threshold: float
) -> Tuple[float, float]:
    # naive (non-vectorized) coverage and fp computation
    coverage_count = 0
    fp_count = 0
    n_data = len(coverage_results[0])
    for i in range(n_data):
        # simualte inference
        best_est = np.inf
        best_idx = None

        for idx, cr in enumerate(coverage_results):
            est = cr.ests[i] + margins[idx]
            if est < best_est:
                best_est = est
                best_idx = idx
        assert best_idx is not None
        detected_solvable = best_est < threshold
        actually_solvable = coverage_results[best_idx].reals[i] < threshold

        if detected_solvable:
            coverage_count += 1
            if not actually_solvable:
                fp_count += 1

    coverage_rate = coverage_count / n_data
    fp_rate = fp_count / coverage_count
    return coverage_rate, fp_rate


def test_compute_coverage_and_fp():
    def est_model(c, x):
        n_call = int(np.linalg.norm(c - x) * 1000)
        return n_call

    def actual_model(c, x):
        n_call_tmp = est_model(c, x)
        tmp = n_call_tmp + np.random.randint(n_call_tmp) * 0.5 + 50
        return int(tmp)

    # setup dummy data
    threshold = 500
    n_data = 1000
    X = np.random.rand(n_data, 2)
    centers = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.0]])

    cr_list: List[CoverageResult] = []
    for c in centers:
        est_values = np.array([est_model(c, x) for x in X])
        real_values = np.array([actual_model(c, x) for x in X])
        cr = CoverageResult(real_values, est_values, threshold)
        cr_list.append(cr)

    # test main
    margins = np.array([10, 100, 20, 40, 200])
    coverage_jit, fp_rate_jit = compute_coverage_and_fp_jit(
        margins,
        np.array([cr.reals for cr in cr_list]),
        np.array([cr.ests for cr in cr_list]),
        threshold,
    )

    coverage_rate_truth, fp_rate_truth = compute_coverage_and_fp_naive(margins, cr_list, threshold)
    np.testing.assert_almost_equal(coverage_jit, coverage_rate_truth)
    np.testing.assert_almost_equal(fp_rate_jit, fp_rate_truth)


if __name__ == "__main__":
    test_compute_coverage_and_fp()
