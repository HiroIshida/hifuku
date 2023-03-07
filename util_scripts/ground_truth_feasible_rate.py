import argparse

from hifuku.datagen.batch_sampler import DistributeBatchProblemSampler
from hifuku.datagen.batch_solver import DistributedBatchProblemSolver
from hifuku.pool import TrivialProblemPool
from hifuku.script_utils import DomainSelector

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-domain", type=str, default="tbrr_rrt", help="")
    args = parser.parse_args()
    domain_name: str = args.domain

    domain = DomainSelector[domain_name].value
    task_type = domain.get_task_type()

    n_sample = 5000

    pool = TrivialProblemPool(task_type, 1)
    sampler = DistributeBatchProblemSampler()
    tasks = sampler.sample_batch(n_sample, pool.as_predicated())

    solver_type = domain.get_solver_type()
    config = domain.get_solver_config()
    config.n_max_satisfaction_trial = 10
    config.n_max_call = 20000

    solver = DistributedBatchProblemSolver(solver_type, config)
    results = solver.solve_batch(tasks, [None] * n_sample)

    success_count = sum([res[0].traj is not None for res in results])
    print(success_count / n_sample)
