import warnings

import numpy as np
from mohou.file import get_project_path

from hifuku.datagen import DistributeBatchProblemSampler, MultiProcessBatchProblemSolver
from hifuku.pool import SimpleIteratorProblemPool
from hifuku.threedim.tabletop import TabletopMeshProblem
from hifuku.utils import create_default_logger

warnings.filterwarnings("ignore", message="Values in x were outside bounds during")
warnings.filterwarnings("ignore", message="texture specified in URDF is not supported")

np.random.seed(0)

if __name__ == "__main__":
    n_problem = 1000
    pp = get_project_path("tabletop_mesh")
    logger = create_default_logger(pp, "mesh_generation")
    cache_base_path = pp / "cache"
    cache_base_path.mkdir(exist_ok=True, parents=True)
    sampler = DistributeBatchProblemSampler[TabletopMeshProblem]()
    pool = SimpleIteratorProblemPool(TabletopMeshProblem, n_problem_inner=0)
    problems = sampler.sample_batch(n_problem, pool.as_predicated())
    solver = MultiProcessBatchProblemSolver[TabletopMeshProblem]()  # this doesn't actually solve
    solver.create_dataset(problems, [np.zeros(0)] * len(problems), cache_base_path, None)
