import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hifuku.domain import HumanoidTableRarmReaching_SQP_Domain

domain = HumanoidTableRarmReaching_SQP_Domain

result_base_path = Path("./result")
result_path: Path = result_base_path / "result-{}".format(domain.get_domain_name())
with result_path.open(mode="rb") as f:
    result_tables = pickle.load(f)

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
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    keys = plot_table_success.keys()

    times_list = []
    for i, key in enumerate(keys):
        value = plot_table_success[key]
        times = [v[1] for v in value]
        times_list.append(times)
    ax.boxplot(times_list, showfliers=False, medianprops={"color": "k"})

    for i, (times, key) in enumerate(zip(times_list, keys)):
        jitter = np.random.normal(scale=0.1, size=len(times))
        ax.scatter(i + 1 + jitter, times, alpha=0.5, s=3, label=key)
    # ax.legend()
    ax.set_xticks(np.arange(1, len(keys) + 1))
    ax.set_xticklabels(list(keys), rotation="vertical")

    fig.subplots_adjust(bottom=0.3)
    plt.show()
