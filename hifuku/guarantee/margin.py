from dataclasses import dataclass
from functools import cached_property
from math import floor

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
            self.values_ground_truth < self.threshold, self.values_estimation < self.threshold
        )

    @cached_property
    def true_negative_bools(self) -> np.ndarray:
        return np.logical_and(
            self.values_ground_truth > self.threshold, self.values_estimation > self.threshold
        )

    @cached_property
    def false_postive_bools(self) -> np.ndarray:
        return np.logical_and(
            self.values_ground_truth > self.threshold, self.values_estimation < self.threshold
        )

    @cached_property
    def false_negative_bools(self) -> np.ndarray:
        return np.logical_and(
            self.values_ground_truth < self.threshold, self.values_estimation > self.threshold
        )

    def determine_margin(self, acceptable_fales_positive_rate: float) -> float:
        fp_bools = self.false_postive_bools
        values_est_fp = self.values_estimation[fp_bools]
        diffs = self.threshold - values_est_fp
        assert np.all(diffs > 0.0)

        num_acceptable = floor(len(self) * acceptable_fales_positive_rate)

        n_fp = sum(fp_bools)
        is_already_satisifed = n_fp <= num_acceptable
        if is_already_satisifed:
            return 0.0

        sorted_diffs = np.sort(diffs)[::-1]  # decreasing order
        return sorted_diffs[num_acceptable] + 1e-6
