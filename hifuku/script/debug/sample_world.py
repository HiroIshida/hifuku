import warnings

import numpy as np
from mohou.file import get_project_path

from hifuku.datagen import MultiProcessDatasetGenerator
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
    gen = MultiProcessDatasetGenerator(TabletopPlanningProblem)

    problem = TabletopPlanningProblem.create_standard()
    result = problem.solve()[0]
    assert result.success
    gen.generate(result.x, n_problem, 50, cache_base_path)
