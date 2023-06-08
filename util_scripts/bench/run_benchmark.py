import argparse
import pickle
from pathlib import Path

import torch
from mohou.file import get_project_path
from rpbench.compatible_solver import CompatibleSolvers

from hifuku.domain import select_domain
from hifuku.library import LibraryBasedSolver, SolutionLibrary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-domain", type=str, default="humanoid_trr_sqp", help="")
    parser.add_argument("-n", type=int, default=300, help="")
    args = parser.parse_args()

    n_sample = args.n

    domain_name: str = args.domain
    domain = select_domain(domain_name)
    pp = get_project_path(domain_name)
    task_type = domain.task_type

    solver_table = CompatibleSolvers.get_compatible_solvers(task_type.__name__)

    # setup proposed solver
    pp = get_project_path("hifuku-{}".format(domain.get_domain_name()))
    libraries = SolutionLibrary.load(pp, domain.task_type, domain.solver_type, torch.device("cpu"))
    lib = libraries[0]
    proposed = LibraryBasedSolver.init(lib)
    solver_table["proposed"] = proposed

    results = []
    false_positive_seq = []

    for i in range(n_sample):
        print(i)

        task = task_type.sample(1)

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
