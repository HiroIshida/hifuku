import argparse
import pickle
from pathlib import Path

from hifuku.datagen.batch_sampler import (
    DistributeBatchProblemSampler,
    MultiProcessBatchProblemSampler,
)
from hifuku.datagen.batch_solver import (
    DistributedBatchProblemSolver,
    MultiProcessBatchProblemSolver,
)
from hifuku.domain import select_domain
from hifuku.pool import TrivialProblemPool
from hifuku.utils import create_default_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=300, help="solved problem number")
    parser.add_argument("-batch", type=int, default=400, help="solved problem number")
    parser.add_argument("-domain", type=str, default="humanoid_trr_sqp", help="")
    parser.add_argument("-mode", type=str, default="lib", help="")
    parser.add_argument("--distributed", action="store_true", help="use distributed")
    args = parser.parse_args()

    domain_name: str = args.domain
    domain = select_domain(domain_name)
    task_type = domain.task_type

    args = parser.parse_args()
    n_data = args.n
    use_distributed = args.distributed
    mode: str = args.mode
    assert mode in ["lib", "pset"]

    logger = create_default_logger(Path("./"), "create_library")

    pool = TrivialProblemPool(task_type, 1)
    if use_distributed:
        sampler = DistributeBatchProblemSampler()
        solver = DistributedBatchProblemSolver(None, None)
    else:
        sampler = MultiProcessBatchProblemSampler()
        solver = MultiProcessBatchProblemSolver(None, None)

    pairs = []
    n_task_batch = args.batch
    n_sample_count = 0
    while len(pairs) < n_data:
        logger.info("new loop")
        tasks = sampler.sample_batch(n_task_batch, pool.as_predicated(), delete_cache=True)
        resultss = solver.solve_batch(tasks, [None] * n_task_batch, use_default_solver=True)
        for task, results in zip(tasks, resultss):
            if results[0].traj is not None:
                pairs.append((task, results[0].traj))
        n_sample_count += n_task_batch
        logger.info("num feasible: {}".format(len(pairs)))
        logger.info("feasible rate: {}".format(len(pairs) / n_sample_count))

    if mode == "lib":
        p = Path("./raw_library/{}.pkl".format(task_type.__name__))
    else:
        p = Path("./problem_set/{}.pkl".format(task_type.__name__))

    logger.info("dump {} pairs to {}".format(len(pairs), p))
    with p.open(mode="wb") as f:
        pickle.dump(pairs, f)
