import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from hifuku.domain import HumanoidTableRarmReaching_SQP_Domain

domain = HumanoidTableRarmReaching_SQP_Domain

result_base_path = Path("./result")
result_path: Path = result_base_path / "result-{}".format(domain.get_domain_name())
with result_path.open(mode="rb") as f:
    result_tables = pickle.load(f)

plot_table = {}
for problem_idx, result_table in enumerate(result_tables):
    for key, value in result_table.items():
        if key not in plot_table:
            plot_table[key] = []
        if value.traj is not None:
            elapsed = value.time_elapsed
            plot_table[key].append((problem_idx, elapsed))
        else:
            pass

fig, ax = plt.subplots()
for key, value in plot_table.items():
    indices, elapsed_times = zip(*value)
    print(key, len(indices))
    ax.scatter(indices, elapsed_times, label=key)
ax.legend()
plt.show()
