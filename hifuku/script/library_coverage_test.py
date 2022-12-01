from pathlib import Path

import numpy as np
import torch

from hifuku.datagen import MultiProcessBatchProblemSolver
from hifuku.library import SolutionLibrary
from hifuku.threedim.tabletop import TabletopPlanningProblem

np.random.seed(0)
p = Path("~/.mohou/tabletop_solution_library").expanduser()
validation_pool = [TabletopPlanningProblem.sample(1) for _ in range(100)]
lib = SolutionLibrary.load(p, TabletopPlanningProblem, device=torch.device("cpu"))[0]
solver = MultiProcessBatchProblemSolver[TabletopPlanningProblem]()
coverage = lib.measure_full_coverage(validation_pool, solver)
print(coverage)
