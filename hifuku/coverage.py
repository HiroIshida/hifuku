import copy
import json
import logging
from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Tuple

import numpy as np
import tqdm
from cmaes import CMA
from numba import jit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoverageResult:
    reals: np.ndarray
    ests: np.ndarray
    threshold: float

    def dumps(self) -> str:
        d = {
            "reals": self.reals.tolist(),
            "ests": self.ests.tolist(),
            "threshold": self.threshold,
        }
        return json.dumps(d)

    @classmethod
    def loads(cls, s: str) -> "CoverageResult":
        d = json.loads(s)
        reals = np.array(d["reals"])
        ests = np.array(d["ests"])
        threshold = d["threshold"]
        return cls(reals, ests, threshold)

    def __post_init__(self):
        assert len(self.reals) == len(self.ests)
        self.reals.flags.writeable = False
        self.ests.flags.writeable = False

    def __len__(self) -> int:
        return len(self.reals)

    def bootstrap_sampling(self) -> "CoverageResult":
        n = self.__len__()
        indices = np.random.randint(n, size=n)
        vgt = self.reals[indices]
        vest = self.ests[indices]
        return CoverageResult(vgt, vest, self.threshold)

    @cached_property
    def true_positive_bools(self) -> np.ndarray:
        return np.logical_and(self.reals <= self.threshold, self.ests <= self.threshold)

    @cached_property
    def true_negative_bools(self) -> np.ndarray:
        return np.logical_and(self.reals > self.threshold, self.ests > self.threshold)

    @cached_property
    def false_postive_bools(self) -> np.ndarray:
        return np.logical_and(self.reals > self.threshold, self.ests <= self.threshold)

    @cached_property
    def false_negative_bools(self) -> np.ndarray:
        return np.logical_and(self.reals <= self.threshold, self.ests > self.threshold)

    def compute_coverage_rate(self, margin: float, eps: float = 1e-6) -> float:
        positive_est = self.ests + margin + eps < self.threshold
        n_positive = np.sum(positive_est)
        return float(n_positive / len(self.ests))

    def compute_false_positive_rate(self, margin: float, eps: float = 1e-6) -> Optional[float]:
        positive_est = self.ests + margin + eps < self.threshold
        n_positive = np.sum(positive_est)
        no_positive = n_positive == 0
        if no_positive:
            return None

        positive_gt = self.reals <= self.threshold
        tp_rate = np.sum(np.logical_and(positive_gt, positive_est)) / n_positive
        fp_rate = 1.0 - tp_rate
        return fp_rate

    def determine_margin(self, acceptable_false_positive_rate: float) -> Tuple[float, float]:
        """
        note that fp rate is defined as fp/(fp + tp)
        """
        fp_bools = self.false_postive_bools
        values_est_fp = self.ests[fp_bools]
        diffs = self.threshold - values_est_fp
        assert np.all(diffs >= 0.0)

        rate = self.compute_false_positive_rate(0.0)
        if rate is None:
            # make no sense to set margin because no positive est
            return np.inf, 0.0

        if rate < acceptable_false_positive_rate:
            # no need to set margin
            return 0.0, self.compute_coverage_rate(0.0)

        sorted_diffs = np.sort(diffs)
        for i in range(len(diffs)):
            margin_cand = sorted_diffs[i]
            rate = self.compute_false_positive_rate(margin_cand)
            logger.debug("margin_cand: {}, fp_rate: {}".format(margin_cand, rate))
            if rate is None:
                return np.inf, 0.0
            if rate < acceptable_false_positive_rate:
                margin_final = margin_cand + 1e-6
                return margin_final, self.compute_coverage_rate(margin_final)
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


@jit(nopython=True)
def compute_coverage_and_fp_jit(
    margins: np.ndarray,
    realss: np.ndarray,
    estss: np.ndarray,
    threshold: float,
) -> Tuple[float, float]:

    N_path, N_mc = realss.shape
    n_coverage = 0
    n_fp = 0
    for j in range(N_mc):
        best_path_idx = np.argmin(estss[:, j] + margins)
        is_est_ok = estss[best_path_idx, j] + margins[best_path_idx] < threshold
        is_real_ok = realss[best_path_idx, j] < threshold
        if is_est_ok:
            n_coverage += 1
            if not is_real_ok:
                n_fp += 1
    coverage_rate = n_coverage / N_mc
    if n_coverage > 0:
        fp_rate = n_fp / n_coverage
    else:
        fp_rate = np.inf
    return coverage_rate, fp_rate


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

    realss = np.array([cr.reals for cr in coverage_results], order="F")
    estss = np.array([cr.ests for cr in coverage_results], order="F")

    optimizer = CMA(mean=margins_guess, sigma=cma_sigma)
    for generation in tqdm.tqdm(range(1000)):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            coverage_est, fp_rate = compute_coverage_and_fp_jit(x, realss, estss, threshold)
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
        if optimizer.should_stop():
            break

    coverage_est_cand, fp_rate_cand = compute_coverage_and_fp_jit(
        best_margins, realss, estss, threshold
    )
    logger.info("[cma result] coverage: {}, fp: {}".format(coverage_est_cand, fp_rate_cand))
    if coverage_est_cand > minimum_coverage and fp_rate_cand < target_fp_rate:
        logger.info("cma result accepted")
        return DetermineMarginsResult(list(best_margins), coverage_est_cand, fp_rate_cand)
    else:
        logger.info("cma result rejected. Returning None")
        return None
