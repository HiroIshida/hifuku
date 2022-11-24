from pathlib import Path

import torch

from hifuku.library import SimpleFixedProblemPool, SolutionLibrary
from hifuku.threedim.tabletop import TabletopPlanningProblem

p = Path("~/.mohou/tabletop_solution_library").expanduser()
validation_pool = SimpleFixedProblemPool.initialize(TabletopPlanningProblem, 2000)
lib = SolutionLibrary.load(p, TabletopPlanningProblem, device=torch.device("cpu"))[0]
coverage = lib.measure_full_coverage(validation_pool)
print(coverage)
