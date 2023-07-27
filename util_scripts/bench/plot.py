import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from skmp.solver.interface import ResultProtocol

from hifuku.domain import select_domain

COLORS = ["blue", "green", "orange", "red"]


@dataclass
class ResultsPerSolver:
    results: List[ResultProtocol]

    def __len__(self) -> int:
        return len(self.results)

    def success_rate(self, timeout: Optional[float] = None) -> float:
        c = 0
        for r in self.results:
            if r.traj is not None:
                if timeout is None or r.time_elapsed < timeout:
                    c += 1
        return c / len(self)

    def times_success(self, timeout: Optional[float] = None) -> np.ndarray:
        times = []
        for r in self.results:
            if r.traj is not None:
                if timeout is None or r.time_elapsed < timeout:
                    times.append(r.time_elapsed)
        return np.array(times)


def convert_list_of_tables_to_table_of_list(
    tables: List[Dict[str, ResultProtocol]]
) -> Dict[str, ResultsPerSolver]:
    table_of_lists = {}

    for table in tables:
        for key, value in table.items():
            if key not in table_of_lists:
                table_of_lists[key] = []
            table_of_lists[key].append(value)

    out = {}
    for key, value in table_of_lists.items():
        out[key] = ResultsPerSolver(value)
    return out


def plot_success_rate(ax, results_table: Dict[str, ResultsPerSolver]):
    width = 0.3
    eps = 0.02
    for i, key in enumerate(results_table.keys()):
        color = COLORS[i]
        pos = i + 1
        if key == "rrtconnect":
            ax.bar(pos - width - eps, results_table[key].success_rate(5), width=width, color=color)
            ax.bar(pos + 0.0, results_table[key].success_rate(10), width=width, color=color)
            ax.bar(pos + width + eps, results_table[key].success_rate(20), width=width, color=color)
        else:
            ax.bar(pos, results_table[key].success_rate(None), width=width, color=color)
    ax.yaxis.grid(color="gray", linestyle="dashed")
    ax.set_ylabel("success rate [-]")
    ax.set_xticks([], minor=False)


def plot_speed_comparison(ax, results_table: Dict[str, ResultsPerSolver], logscale: bool):

    for i, key in enumerate(results_table.keys()):

        if key == "rrtconnect":
            times_list = [results_table[key].times_success(to) for to in [5, 10, 20]]
            positions = [i + 1 + slide for slide in [-0.3, 0.0, 0.3]]
            ax.boxplot(
                times_list, positions=positions, showfliers=False, medianprops={"color": "k"}
            )
        else:
            times = results_table[key].times_success()
            ax.boxplot(times, positions=[i + 1], showfliers=False, medianprops={"color": "k"})

        times = results_table[key].times_success()
        jitter = np.random.normal(scale=0.15, size=len(times))
        ax.scatter(i + 1 + jitter, times, alpha=0.3, s=3, label=key, c=COLORS[i])

    if logscale:
        ax.set_yscale("log")
        ax.set_ylabel("elapsed time \n in logscale [s]")
        ax.yaxis.grid(which="major", color="gray", linestyle="-", alpha=1.0)
        ax.yaxis.grid(which="minor", color="gray", linestyle="dashed", alpha=0.5)
    else:
        ax.set_ylabel("elapsed time [s]")
        ax.yaxis.grid(color="gray", linestyle="dashed")
    ax.set_ylim([1e-1, 30.0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=300, help="")
    parser.add_argument("-m", type=int, default=12, help="number of process")
    parser.add_argument("-domain", type=str, default="humanoid_trr_sqp", help="")
    parser.add_argument("--log", action="store_true", help="log plot")
    parser.add_argument("--save", action="store_true", help="save")
    parser.add_argument("-dpi", type=int, default=500)
    args = parser.parse_args()

    domain_name: str = args.domain
    domain = select_domain(domain_name)
    result_base_path = Path("./result")
    result_path: Path = result_base_path / "result-{}".format(domain.get_domain_name())

    result_tables: List[Dict[str, ResultProtocol]]
    with result_path.open(mode="rb") as f:
        tasks, result_tables = pickle.load(f)
    results_table = convert_list_of_tables_to_table_of_list(result_tables)

    fig, axes = plt.subplots(2, sharex=True)
    plot_success_rate(axes[0], results_table)
    plot_speed_comparison(axes[1], results_table, args.log)

    if args.save:
        parent = Path("./figs/").expanduser()
        parent.mkdir(exist_ok=True)
        file_path = parent / "comparison-{}.png".format(domain_name)
        plt.savefig(file_path, dpi=args.dpi)
    else:
        plt.show()
