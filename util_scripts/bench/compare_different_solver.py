import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hifuku.domain import select_domain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=300, help="")
    parser.add_argument("-m", type=int, default=12, help="number of process")
    parser.add_argument("-domain", type=str, default="humanoid_trr_sqp", help="")

    args = parser.parse_args()

    domain_name: str = args.domain
    domain = select_domain(domain_name)

    result_base_path = Path("./result")
    result_path: Path = result_base_path / "result-{}".format(domain.get_domain_name())
    with result_path.open(mode="rb") as f:
        result_tables, _ = pickle.load(f)

    plot_table_success = {}
    plot_table_failure = {}
    for problem_idx, result_table in enumerate(result_tables):
        for key, value in result_table.items():
            if key not in plot_table_success:
                plot_table_success[key] = []
                plot_table_failure[key] = []
            if value.traj is not None:
                plot_table_success[key].append((problem_idx, value.time_elapsed))
            else:
                plot_table_failure[key].append((problem_idx, value.time_elapsed))

    success_rate_table = {}
    for key in plot_table_success.keys():
        n_total = len(plot_table_success[key]) + len(plot_table_failure[key])
        n_success = len(plot_table_success[key])
        success_rate_table[key] = n_success / n_total
    print(success_rate_table)

    instance_wise = False
    if instance_wise:
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        for key, value in plot_table_success.items():
            indices, elapsed_times = zip(*value)
            print(key, len(indices))
            ax.scatter(indices, elapsed_times, label=key)
        ax.legend()
        plt.show()
    else:
        keys = plot_table_success.keys()

        fig, axes = plt.subplots(2)
        ax1 = axes[0]
        ax1.bar(np.arange(1, len(keys) + 1), success_rate_table.values(), width=0.5)
        ax1.set_ylim([0, 1.0])
        ax1.grid()
        ax1.set_ylabel("success rate")
        ax1.set_xticks([], minor=False)

        ax2 = axes[1]
        ax2.set_yscale("log")
        ax2.grid()
        ax2.set_ylabel("planning time")

        times_list = []
        for i, key in enumerate(keys):
            value = plot_table_success[key]
            times = [v[1] for v in value]
            times_list.append(times)
        ax2.boxplot(times_list, showfliers=False, medianprops={"color": "k"})

        for i, (times, key) in enumerate(zip(times_list, keys)):
            jitter = np.random.normal(scale=0.1, size=len(times))
            ax2.scatter(i + 1 + jitter, times, alpha=0.5, s=3, label=key)
        ax2.set_xticks(np.arange(1, len(keys) + 1))
        ax2.set_xticklabels(list(keys), rotation=45)

        fig.subplots_adjust(bottom=0.3)
        plt.show()
