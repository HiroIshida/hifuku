import copy
import logging
from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Tuple

import numpy as np
import tqdm
from cmaes import CMA

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoverageResult:
    values_ground_truth: np.ndarray
    values_estimation: np.ndarray
    threshold: float

    def __post_init__(self):
        assert len(self.values_ground_truth) == len(self.values_estimation)
        self.values_ground_truth.flags.writeable = False
        self.values_estimation.flags.writeable = False

    def __len__(self) -> int:
        return len(self.values_ground_truth)

    def bootstrap_sampling(self) -> "CoverageResult":
        n = self.__len__()
        indices = np.random.randint(n, size=n)
        vgt = self.values_ground_truth[indices]
        vest = self.values_estimation[indices]
        return CoverageResult(vgt, vest, self.threshold)

    @cached_property
    def true_positive_bools(self) -> np.ndarray:
        return np.logical_and(
            self.values_ground_truth <= self.threshold, self.values_estimation <= self.threshold
        )

    @cached_property
    def true_negative_bools(self) -> np.ndarray:
        return np.logical_and(
            self.values_ground_truth > self.threshold, self.values_estimation > self.threshold
        )

    @cached_property
    def false_postive_bools(self) -> np.ndarray:
        return np.logical_and(
            self.values_ground_truth > self.threshold, self.values_estimation <= self.threshold
        )

    @cached_property
    def false_negative_bools(self) -> np.ndarray:
        return np.logical_and(
            self.values_ground_truth <= self.threshold, self.values_estimation > self.threshold
        )

    def compute_false_positive_rate(self, margin: float, eps: float = 1e-6) -> Optional[float]:
        positive_est = self.values_estimation + margin + eps < self.threshold
        n_positive = np.sum(positive_est)
        no_positive = n_positive == 0
        if no_positive:
            return None

        positive_gt = self.values_ground_truth <= self.threshold
        tp_rate = np.sum(np.logical_and(positive_gt, positive_est)) / n_positive
        fp_rate = 1.0 - tp_rate
        return fp_rate

    def determine_margin(self, acceptable_false_positive_rate: float) -> float:
        """
        note that fp rate is defined as fp/(fp + tp)
        """
        fp_bools = self.false_postive_bools
        values_est_fp = self.values_estimation[fp_bools]
        diffs = self.threshold - values_est_fp
        assert np.all(diffs >= 0.0)

        rate = self.compute_false_positive_rate(0.0)
        if rate is None:
            # make no sense to set margin because no positive est
            return np.inf

        if rate < acceptable_false_positive_rate:
            # no need to set margin
            return 0.0

        sorted_diffs = np.sort(diffs)
        for i in range(len(diffs)):
            margin_cand = sorted_diffs[i]
            rate = self.compute_false_positive_rate(margin_cand)
            logger.debug("margin_cand: {}, fp_rate: {}".format(margin_cand, rate))
            if rate is None:
                return np.inf
            if rate < acceptable_false_positive_rate:
                margin_final = margin_cand + 1e-6
                return margin_final
        assert False, "final rate {}".format(rate)

    def __str__(self) -> str:
        string = "coverage result => "
        string += "n_sample: {}, ".format(len(self))
        string += "true positive: {}, ".format(sum(self.true_positive_bools))
        string += "true negative: {}, ".format(sum(self.true_negative_bools))
        string += "false positive: {}, ".format(sum(self.false_postive_bools))
        string += "false negative: {}".format(sum(self.false_negative_bools))
        return string


@dataclass
class DetermineMarginsResult:
    best_margins: List[float]
    coverage: float
    fprate: float


def compute_coverage_and_fp(
    margins: np.ndarray, coverage_results: List[CoverageResult], threshold: float
) -> Tuple[float, float]:

    est_arr_list, real_arr_list = [], []
    for coverage_result, margin in zip(coverage_results, margins):
        est_arr_list.append(coverage_result.values_estimation + margin)
        real_arr_list.append(coverage_result.values_ground_truth)
    est_mat = np.array(est_arr_list)
    real_mat = np.array(real_arr_list)

    # find element indices which have smallest est value
    est_arr = np.min(est_mat, axis=0)
    element_indices = np.argmin(est_mat, axis=0)
    real_arr = np.array([real_mat[idx, i] for i, idx in enumerate(element_indices)])

    # compute (true + false) positive values
    est_bool_arr = est_arr < threshold
    real_bool_arr = real_arr < threshold

    n_total = len(est_arr)
    coverage_est = sum(est_bool_arr) / n_total

    # compute false positve rate
    fp_rate = sum(np.logical_and(est_bool_arr, ~real_bool_arr)) / sum(est_bool_arr)

    return coverage_est, fp_rate


def determine_margins(
    coverage_results: List[CoverageResult],
    threshold: float,
    target_fp_rate: float,
    cma_sigma: float,
    margins_guess: Optional[np.ndarray] = None,
    minimum_coverage: Optional[float] = None,
) -> Optional[DetermineMarginsResult]:

    target_fp_rate_modified = target_fp_rate - 1e-3  # because penalty method is not tight
    logger.debug("target fp_rate modified: {}".format(target_fp_rate_modified))

    if margins_guess is None:
        n_pred = len(coverage_results)
        margins_guess = np.zeros(n_pred)

    best_score = np.inf
    best_margins = copy.deepcopy(margins_guess)

    if minimum_coverage is None:
        minimum_coverage = 0.0

    coverage_est = -np.inf
    fp_rate = -np.inf

    optimizer = CMA(mean=margins_guess, sigma=cma_sigma)
    for generation in tqdm.tqdm(range(500)):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            coverage_est, fp_rate = compute_coverage_and_fp(x, coverage_results, threshold)
            J = -coverage_est + 1e4 * max(fp_rate - target_fp_rate_modified, 0) ** 2
            solutions.append((x, J))
        optimizer.tell(solutions)

        xs, values = zip(*solutions)
        best_index = np.argmin(values)

        if values[best_index] < best_score:
            best_score = values[best_index]
            best_margins = xs[best_index]

        logger.debug(
            "[generation {}] coverage: {}, fp_rate: {}".format(generation, coverage_est, fp_rate)
        )

    coverage_est_cand, fp_rate_cand = compute_coverage_and_fp(
        best_margins, coverage_results, threshold
    )
    logger.info("[cma result] coverage: {}, fp: {}".format(coverage_est_cand, fp_rate_cand))
    if coverage_est_cand > minimum_coverage and fp_rate_cand < target_fp_rate_modified:
        logger.info("cma result accepted")
        return DetermineMarginsResult(list(best_margins), coverage_est_cand, fp_rate_cand)
    else:
        logger.info("cma result rejected. Returning None")
        return None
