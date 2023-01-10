import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
import tqdm
from mohou.file import get_project_path
from ompl import LightningDB
from rpbench.interface import SkmpTaskSolver
from skmp.solver.ompl_solver import LightningSolver, OMPLSolver, OMPLSolverConfig

from hifuku.domain import TBRR_SQP_DomainProvider
from hifuku.library import LibraryBasedSolver, SolutionLibrary

pp = get_project_path(
    "tabletop_solution_library-{}".format(TBRR_SQP_DomainProvider.get_domain_name())
)
task_type = TBRR_SQP_DomainProvider.get_task_type()
libraries = SolutionLibrary.load(pp, task_type, OMPLSolver, torch.device("cpu"))
hifuku = LibraryBasedSolver.init(libraries[0])

ompl_solcon = OMPLSolverConfig(n_max_call=2000, n_max_satisfaction_trial=30)
rrt_connect = SkmpTaskSolver.init(OMPLSolver.init(ompl_solcon), task_type)

db = LightningDB(10)
db.load("./lightning.db")
LightningSolver.init(ompl_solcon, db)
lightning = SkmpTaskSolver.init(LightningSolver.init(ompl_solcon, db), task_type)


@dataclass
class Result:
    success: bool
    time: float


names = ["rrt_connect", "lightning", "hifuku"]
solvers = [rrt_connect, lightning, hifuku]
resultss = [[], [], []]

n_test = 30
results: List[Result] = []
for _ in tqdm.tqdm(range(n_test)):
    task = task_type.sample(1)
    for solver, results in zip(solvers, resultss):
        solver.setup(task)
        res = solver.solve()
        success = res.traj is not None
        print(success)
        results.append(Result(success, res.time_elapsed))

with Path("/tmp/bench.pkl").open(mode="wb") as f:
    pickle.dump(resultss, f)

with Path("/tmp/bench.pkl").open(mode="rb") as f:
    resultss = pickle.load(f)

fig, ax = plt.subplots()
for name, results in zip(names, resultss):
    times = [r.time for r in results]
    ax.plot(times, label=name)
ax.legend()
plt.show()
