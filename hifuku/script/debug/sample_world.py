import warnings

import numpy as np
from mohou.file import get_project_path

from hifuku.datagen import MultiProcessBatchProblemSolver
from hifuku.threedim.tabletop import TabletopPlanningProblem

warnings.filterwarnings("ignore", message="Values in x were outside bounds during")
warnings.filterwarnings("ignore", message="texture specified in URDF is not supported")

np.random.seed(0)

if __name__ == "__main__":
    n_problem = 100
    pp = get_project_path("tabletop_ik")
    pp.mkdir(exist_ok=True)
    cache_base_path = pp / "cache"
    cache_base_path.mkdir()
    gen = MultiProcessBatchProblemSolver[TabletopPlanningProblem]()

    x_init = TabletopPlanningProblem.get_default_init_solution()
    problems = [TabletopPlanningProblem.sample(50) for _ in range(n_problem)]
    init_solutions = [x_init] * n_problem
    gen.create_dataset(problems, init_solutions, cache_base_path, None)
