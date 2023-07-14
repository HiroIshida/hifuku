import argparse
import pickle
from pathlib import Path

import torch
from mohou.file import get_project_path
from rpbench.compatible_solver import CompatibleSolvers

from hifuku.domain import select_domain
from hifuku.library import (
    LibraryBasedGuaranteedSolver,
    LibraryBasedHeuristicSolver,
    SolutionLibrary,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-domain", type=str, default="humanoid_trr_sqp", help="")
    parser.add_argument("-n", type=int, default=300, help="")
    parser.add_argument("--feasible", action="store_true", help="use only feasible problem set")
    args = parser.parse_args()

    n_sample = args.n
    only_feasible: bool = args.feasible
    domain_name: str = args.domain
    domain = select_domain(domain_name)
    pp = get_project_path(domain_name)
    task_type = domain.task_type

    if only_feasible:
        print("use feasible problem set")
        problem_set_path = Path("./problem_set") / (task_type.__name__ + ".pkl")
        with problem_set_path.open(mode="rb") as f:
            pairs = pickle.load(f)
            tasks_full, resultss = zip(*pairs)
            for task, results in zip(tasks_full, resultss):
                assert results[0].traj is not None
        assert n_sample <= len(tasks_full)
        tasks = tasks_full[:n_sample]
    else:
        tasks = [task_type.sample(1) for _ in range(n_sample)]
    assert tasks[0].n_inner_task == 1

    solver_table = CompatibleSolvers.get_compatible_solvers(task_type.__name__)

    # setup proposed solver
    pp = get_project_path("hifuku-{}".format(domain.get_domain_name()))
    libraries = SolutionLibrary.load(pp, domain.task_type, domain.solver_type, torch.device("cpu"))
    lib = libraries[0]
    proposed = LibraryBasedGuaranteedSolver.init(lib)
    solver_table["proposed-guaranteed"] = proposed

    proposed = LibraryBasedHeuristicSolver.init(lib)
    solver_table["proposed-heuristic"] = proposed

    results = []
    false_positive_seq = []

    for i, task in enumerate(tasks):
        print(i)

        result = {}
        for name, solver in solver_table.items():
            print("solver name: {}".format(name))

            print("start setting up")
            solver.setup(task)

            print("start solving")
            res = solver.solve()
            print("solved?: {}, time: {}".format(res.traj is not None, res.time_elapsed))

            result[name] = res

            if name == "proposed":
                if solver.previous_false_positive is not None:
                    false_positive_seq.append(solver.previous_false_positive)
                    false_positive_count = sum(false_positive_seq)
                    fp_rate = false_positive_count / len(false_positive_seq)
                    print("current fp rate: {}".format(fp_rate))

        results.append(result)

    result_base_path = Path("./result")
    result_path: Path = result_base_path / "result-{}".format(domain.get_domain_name())
    with result_path.open(mode="wb") as f:
        pickle.dump((results, false_positive_seq), f)
