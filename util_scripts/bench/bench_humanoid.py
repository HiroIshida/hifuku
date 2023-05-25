import pickle
from pathlib import Path

import torch
from mohou.file import get_project_path
from rpbench.compatible_solver import CompatibleSolvers
from rpbench.jaxon.below_table import HumanoidTableReachingTask

from hifuku.domain import HumanoidTableRarmReaching_SQP_Domain
from hifuku.library import LibraryBasedSolver, SolutionLibrary

domain = HumanoidTableRarmReaching_SQP_Domain
solver_table = CompatibleSolvers.get_compatible_solvers(domain.task_type)

# setup proposed solver
pp = get_project_path("tabletop_solution_library-{}".format(domain.get_domain_name()))
libraries = SolutionLibrary.load(pp, domain.task_type, domain.solver_type, torch.device("cpu"))
lib = libraries[0]
proposed = LibraryBasedSolver.init(lib)
solver_table["proposed"] = proposed

results = []
false_positive_seq = []

for i in range(500):
    print(i)

    task = HumanoidTableReachingTask.sample(1)

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
