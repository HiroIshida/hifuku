from multiprocessing import Process, Queue

import tqdm
from ompl import LightningDB
from rpbench.tabletop import TabletopBoxRightArmReachingTask
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig

from hifuku.datagen import split_number


def split_number(num, div):
    return [num // div + (1 if x < num % div else 0) for x in range(div)]


def worker(idx_process: int, n_sample: int, solver_config: OMPLSolverConfig, out_queue: Queue):
    solver = OMPLSolver.init(solver_config)
    count = 0
    disable = idx_process != 0
    with tqdm.tqdm(total=n_sample, smoothing=0.0, disable=disable) as pbar:
        while count < n_sample:
            task = TabletopBoxRightArmReachingTask.sample(1)
            problem = task.export_problems()[0]
            solver.setup(problem)
            res = solver.solve()
            if res.traj is not None:
                out_queue.put(res.traj)
                count += 1
                pbar.update()


q = Queue()
n_process = 12
n_sample_total = 300
ps = []
n_sample_list = split_number(n_sample_total, n_process)
for i, n_sample in enumerate(n_sample_list):
    args = (i, n_sample, OMPLSolverConfig(10000, 30), q)
    p = Process(target=worker, args=args)
    p.start()
    ps.append(p)

trajectories = [q.get() for _ in range(n_sample_total)]

for p in ps:
    p.join()

db = LightningDB(TabletopBoxRightArmReachingTask.get_dof())
for traj in trajectories:
    db.add_experience(list(traj.numpy()))

db.save("./lightning.db")
