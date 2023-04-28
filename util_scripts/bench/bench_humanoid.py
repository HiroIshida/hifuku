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
solver_table.pop("rrtconnect")

# setup proposed solver
pp = get_project_path("tabletop_solution_library-{}".format(domain.get_domain_name()))
libraries = SolutionLibrary.load(pp, domain.task_type, domain.solver_type, torch.device("cpu"))
lib = libraries[0]
proposed = LibraryBasedSolver.init(lib)
solver_table["proposed"] = proposed

results = []

for i in range(100):
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
    results.append(result)

result_base_path = Path("./result")
result_path: Path = result_base_path / "result-{}".format(domain.get_domain_name())
with result_path.open(mode="wb") as f:
    pickle.dump(results, f)
