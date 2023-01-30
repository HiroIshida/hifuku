from dataclasses import dataclass
from functools import cached_property

import numpy as np


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

    def determine_margin(self, acceptable_false_positive_rate: float) -> float:
        """
        note that fp rate is defined as fp/(fp + tp)
        """
        fp_bools = self.false_postive_bools
        values_est_fp = self.values_estimation[fp_bools]
        diffs = self.threshold - values_est_fp
        assert np.all(diffs > 0.0)

        positive_gt = self.values_ground_truth <= self.threshold

        def fp_rate(margin: float) -> float:
            positive_est = self.values_estimation + margin <= self.threshold
            tp_rate = np.sum(np.logical_and(positive_gt, positive_est)) / np.sum(positive_est)
            fp_rate = 1.0 - tp_rate
            return fp_rate

        if fp_rate(0.0) < acceptable_false_positive_rate:
            return 0.0

        sorted_diffs = np.sort(diffs)
        for i in range(len(diffs)):
            margin_cand = sorted_diffs[i]
            rate = fp_rate(margin_cand)
            if rate < acceptable_false_positive_rate:
                return margin_cand + 1e-5
        assert False

    def __str__(self) -> str:
        string = "coverage result => "
        string += "n_sample: {}, ".format(len(self))
        string += "true positive: {}, ".format(sum(self.true_positive_bools))
        string += "true negative: {}, ".format(sum(self.true_negative_bools))
        string += "false positive: {}, ".format(sum(self.false_postive_bools))
        string += "false negative: {}".format(sum(self.false_negative_bools))
        return string
