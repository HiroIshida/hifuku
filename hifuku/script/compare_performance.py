import pickle
import time
from pathlib import Path

import numpy as np
import torch
import tqdm

from hifuku.datagen import MultiProcessBatchProblemSampler
from hifuku.library import SolutionLibrary
from hifuku.pool import SimpleIteratorProblemPool
from hifuku.threedim.tabletop import TabletopPlanningProblem

sampler = MultiProcessBatchProblemSampler[TabletopPlanningProblem]()
pool = SimpleIteratorProblemPool(TabletopPlanningProblem, 1)
n_problem = 300
problems = sampler.sample_batch(n_problem, pool.as_predicated())

# setup cache
TabletopPlanningProblem.cache_all()

# load lib
p = Path("~/.mohou/tabletop_solution_library").expanduser()
lib = SolutionLibrary.load(p, TabletopPlanningProblem, device=torch.device("cpu"))[0]
lib.limit_thread = False

# solve problem using library
naives = []
mines = []
for prob in tqdm.tqdm(problems):
    ts = time.time()
    res = lib.infer(prob)[0]
    res = prob.solve(res.init_solution)[0]
    mine = time.time() - ts
    if res.success:
        try:
            ts = time.time()
            res = prob.solve()[0]  # from scratch using RRT
            naive = time.time() - ts
        except:
            continue

        if res.success:
            naives.append(naive)
            mines.append(mine)

with open("/tmp/hifuku_bench.pkl", "wb") as f:
    pickle.dump((naives, mines), f)

print("naives")
print(np.mean(naives))
print(np.std(naives))
print(np.max(naives))

print("mines")
print(np.mean(mines))
print(np.std(mines))
print(np.max(mines))
