import argparse

import matplotlib.pyplot as plt
import numpy as np

from hifuku.script_utils import load_library

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-domain", type=str, default="humanoid_tcrr2_sqp", help="")
    args = parser.parse_args()
    domain_name: str = args.domain

    lib = load_library(domain_name, "cpu", postfix="0.1")
    covs = np.hstack([[0.0], np.array(lib._coverage_est_history) * (1 - 0.1)])

    times = [0.0]
    for i, time in enumerate(lib._elapsed_time_history):
        times.append(times[-1] + time / 3600.0)
    times = np.array(times)

    increase_indicse = []
    for i in range(len(covs) - 1):
        inc = covs[i + 1] - covs[i]
        if inc > 1e-6:
            increase_indicse.append(i + 1)
    increase_indicse = np.array(increase_indicse)

    plt.plot(times, covs, "-", color="gray")
    print(increase_indicse)
    plt.plot(times[increase_indicse], covs[increase_indicse], "o")
    plt.show()

    # prev_coverage = 0.0
    # for i, time in enumerate(times):
    #     current_coverage = np.array(lib._coverage_est_history)[i] * (1 - 0.1)
    #     if current_coverage > prev_coverage:
    #         plt.plot(time, current_coverage, 'o', color='b')
    #     else:
    #         plt.plot(time, current_coverage, 'x', color='r')
    #     prev_coverage = current_coverage

    # plt.show()
