import argparse
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from rpbench.tabletop import (
    TabletopBoxDualArmReachingTask,
    TabletopBoxRightArmReachingTask,
)

from hifuku.script_utils import load_library


class TaskSelector(Enum):
    tbrr = TabletopBoxRightArmReachingTask
    tbdr = TabletopBoxDualArmReachingTask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-domain", type=str, default="tbrr_rrt", help="")
    args = parser.parse_args()
    domain_name: str = args.domain

    lib = load_library(domain_name, "cpu")
    n_coverage_sample = len(lib.coverage_results[0])
    bools = np.zeros(n_coverage_sample)

    rates = [0.0]
    for covres, m in zip(lib.coverage_results, lib.margins):
        est_ok = covres.values_estimation + m < covres.threshold
        bools = np.logical_or(bools, est_ok)
        rates.append(np.sum(bools) / n_coverage_sample)
    plt.plot(rates)
    plt.show()
