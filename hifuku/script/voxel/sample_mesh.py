import warnings

import numpy as np
from mohou.file import get_project_path

from hifuku.datagen import MultiProcessBatchProblemSolver
from hifuku.threedim.tabletop import TabletopMeshProblem

warnings.filterwarnings("ignore", message="Values in x were outside bounds during")
warnings.filterwarnings("ignore", message="texture specified in URDF is not supported")

np.random.seed(0)

if __name__ == "__main__":
    n_problem = 20000
    pp = get_project_path("tabletop_mesh")
    cache_base_path = pp / "cache"
    cache_base_path.mkdir(exist_ok=True, parents=True)
    gen = MultiProcessBatchProblemSolver(TabletopMeshProblem)
    gen.generate(np.zeros(0), n_problem, 0, cache_base_path)
