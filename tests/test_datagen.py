import hashlib
import logging
import os
import pickle
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import pytest
from ompl import set_ompl_random_seed
from rpbench.articulated.pr2.minifridge import TabletopClutteredFridgeReachingTask
from skmp.solver.nlp_solver import SQPBasedSolver, SQPBasedSolverConfig
from skmp.trajectory import Trajectory

from hifuku.config import ServerSpec
from hifuku.datagen import (
    BatchProblemSampler,
    BatchProblemSolver,
    DistributeBatchMarginsDeterminant,
    DistributeBatchProblemSampler,
    DistributedBatchProblemSolver,
    MultiProcesBatchMarginsDeterminant,
    MultiProcessBatchProblemSampler,
    MultiProcessBatchProblemSolver,
)
from hifuku.pool import PredicatedProblemPool, ProblemPool
from hifuku.script_utils import create_default_logger
from hifuku.testing_asset import SimplePredicate

logger = logging.getLogger(__name__)

np.random.seed(0)
set_ompl_random_seed(0)
task_type = TabletopClutteredFridgeReachingTask


@pytest.fixture(autouse=True, scope="session")
def server():
    p1 = subprocess.Popen(
        "python3 -m hifuku.datagen.http_datagen.server -port 8081", shell=True, preexec_fn=os.setsid
    )
    p2 = subprocess.Popen(
        "python3 -m hifuku.datagen.http_datagen.server -port 8082", shell=True, preexec_fn=os.setsid
    )
    time.sleep(5)
    yield
    # https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true/4791612#4791612
    p1.terminate()
    p2.terminate()
    p1.wait()
    p2.wait()
    logger.info("killed servers")


@lru_cache(maxsize=1)
def compute_init_traj() -> Trajectory:
    task = task_type.sample(1, standard=True)
    res = task.solve_default()[0]
    assert res.traj is not None
    return res.traj


def test_batch_solver_init_solutions():
    init_traj = compute_init_traj()
    solcon = SQPBasedSolverConfig(
        n_wp=40,
        n_max_call=10,
        motion_step_satisfaction="debug_ignore",
        force_deterministic=True,
    )
    mp_batch_solver = MultiProcessBatchProblemSolver(SQPBasedSolver, solcon, 2)

    n_task = 5
    n_inner = 2
    tasks = [task_type.sample(n_inner) for _ in range(n_task)]
    mp_batch_solver.solve_batch(tasks, [None] * n_task)
    mp_batch_solver.solve_batch(tasks, [init_traj] * n_task)
    mp_batch_solver.solve_batch(tasks, [[init_traj] * n_inner] * n_task)


def test_consistency_of_all_batch_sovler(server):
    init_traj = compute_init_traj()

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        create_default_logger(td_path, "test_datagen")

        for n_problem in [1, 8]:  # to test edge case
            n_problem_inner = 2

            init_solutions = [init_traj] * n_problem
            # set standard = True for testing purpose
            tasks = [task_type.sample(n_problem_inner) for _ in range(n_problem)]
            batch_solver_list: List[BatchProblemSolver] = []

            n_max_call = 10
            solcon = SQPBasedSolverConfig(
                n_wp=20,
                n_max_call=n_max_call,
                motion_step_satisfaction="debug_ignore",
                force_deterministic=True,
            )
            mp_batch_solver = MultiProcessBatchProblemSolver(SQPBasedSolver, solcon, n_process=2)
            assert mp_batch_solver.n_process == 2
            batch_solver_list.append(mp_batch_solver)

            specs = (ServerSpec("localhost", 8081, 1.0), ServerSpec("localhost", 8082, 1.0))
            batch_solver = DistributedBatchProblemSolver(
                SQPBasedSolver, solcon, specs, n_measure_sample=1
            )
            batch_solver_list.append(batch_solver)

            # compare generated nit and success
            # we wanted to directly compare results_list but somehow, pickling-unpickling process
            # change the hash value. so...
            nits_list = []
            successes_list = []
            for batch_solver in batch_solver_list:  # type: ignore
                print(batch_solver)
                results_list = batch_solver.solve_batch(
                    tasks, init_solutions, tmp_n_max_call_mult_factor=1.5
                )
                assert isinstance(results_list, list)
                assert len(results_list) == n_problem
                assert isinstance(results_list[0], tuple)
                assert len(results_list[0]) == n_problem_inner

                nits = []
                successes = []
                for results in results_list:
                    nits.extend([r.n_call for r in results])
                    successes.extend([r.traj is not None for r in results])
                nits_list.append(tuple(nits))
                successes_list.append(tuple(successes))

                # check if multiplicatio by tmp_n_max_call_mult_factor is reset
                assert batch_solver.config.n_max_call == n_max_call

            # NOTE: it seems that osqp solve results sometimes slightly different though
            # the same problem is provided.. maybe random variable is used inside???
            assert len(set(nits_list)) == 1
            assert len(set(successes_list)) == 1


def test_consistency_of_all_batch_sampler(server):
    specs = (ServerSpec("localhost", 8081, 1.0), ServerSpec("localhost", 8082, 1.0))

    sampler_list: List[BatchProblemSampler[task_type]] = []
    sampler_list.append(MultiProcessBatchProblemSampler(1))
    sampler_list.append(DistributeBatchProblemSampler[task_type](specs))

    n_problem_inner = 5
    pool_list: List[PredicatedProblemPool] = []
    pool_base = ProblemPool(task_type, n_problem_inner)
    pool_list.append(pool_base.as_predicated())
    pool_list.append(pool_base.make_predicated(SimplePredicate(), 40))

    for n_sample in [1, 2, 20]:  # to test edge case
        for pool in pool_list:
            for sampler in sampler_list:
                samples = sampler.sample_batch(n_sample, pool)
                assert samples.shape == (n_sample, n_problem_inner, task_type.get_task_dof())

                # in the parallel processing, the typical but difficult-to-find bug is
                # duplication of sample by forgetting to set peroper random seed.
                # check that no duplicate samples here.
                hash_vals = [hashlib.md5(pickle.dumps(s)).hexdigest() for s in samples]
                assert len(set(hash_vals)) == len(hash_vals)

                # Note: because different random seeds are used, the result of the sampling
                # is different. so we cannot compare the result of the sampling directly.


def test_batch_determinant(server):
    coverage_results_path = Path(__file__).resolve().parent / "data" / "coverage_results.pkl"
    with coverage_results_path.open(mode="rb") as f:
        coverage_results = pickle.load(f)

    specs = (ServerSpec("localhost", 8081, 1.0), ServerSpec("localhost", 8082, 1.0))
    determinant_list = []
    determinant_list.append(MultiProcesBatchMarginsDeterminant(4))
    determinant_list.append(DistributeBatchMarginsDeterminant(specs))

    for determinant in determinant_list:
        n_sample = 8
        results = determinant.determine_batch(n_sample, coverage_results, 5, 0.1, 5, None, None)
        assert len(results) == n_sample


if __name__ == "__main__":
    test_create_dataset()
    # test_batch_determinant()
