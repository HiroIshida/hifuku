import numpy as np

from hifuku.coverage import CoverageResult


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


def test_determine_threshold():
    np.random.seed(0)

    threshold = 2.0

    def sample_dataset(n: int = 100000):
        # generate problem instances
        # the problem here is reaching from the origin to some destination.
        # and the cost is distance from the origin with some noise
        positions = np.random.randn(n)
        costs_ground_truth = 2 * np.abs(positions) + np.random.randn(n) * 0.5
        costs_estimation = np.maximum(2 * np.abs(positions) - 1.0, 0)
        return costs_ground_truth, costs_estimation

    costs_ground_truth, costs_estimation = sample_dataset()

    result = CoverageResult(costs_ground_truth, costs_estimation, threshold)
    true_positive_lower_bound = 0.95

    conf_coef = 0.05
    margin = result.determine_margin(true_positive_lower_bound, confidence_coefficient=conf_coef)

    error_count = 0
    for _ in range(1000):
        costs_ground_truth, costs_estimation = sample_dataset()
        result = CoverageResult(costs_ground_truth, costs_estimation + margin, threshold)
        if result.true_positive_rate < true_positive_lower_bound:
            error_count += 1
    error_rate = error_count / 1000
    assert error_rate < conf_coef


if __name__ == "__main__":
    # test_coverage_result()
    test_determine_threshold()
