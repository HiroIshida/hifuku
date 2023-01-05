import logging
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest
from ompl import set_ompl_random_seed
from rpbench.tabletop import TabletopBoxRightArmReachingTask
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig, OMPLSolverResult

from hifuku.config import ServerSpec
from hifuku.datagen import (
    BatchProblemSampler,
    BatchProblemSolver,
    DistributeBatchProblemSampler,
    DistributedBatchProblemSolver,
    MultiProcessBatchProblemSampler,
    MultiProcessBatchProblemSolver,
)
from hifuku.llazy.dataset import LazyDecomplessDataLoader, LazyDecomplessDataset
from hifuku.pool import PredicatedProblemPool, TrivialProblemPool
from hifuku.testing_asset import SimplePredicate
from hifuku.types import RawData
from hifuku.utils import create_default_logger

logger = logging.getLogger(__name__)

np.random.seed(0)
set_ompl_random_seed(0)


@pytest.fixture(autouse=True)
def server():
    p1 = subprocess.Popen(
        "python3 -m hifuku.http_datagen.server -port 8081", shell=True, preexec_fn=os.setsid
    )
    p2 = subprocess.Popen(
        "python3 -m hifuku.http_datagen.server -port 8082", shell=True, preexec_fn=os.setsid
    )
    time.sleep(5)
    yield
    # https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true/4791612#4791612
    os.killpg(os.getpgid(p1.pid), signal.SIGTERM)
    os.killpg(os.getpgid(p2.pid), signal.SIGTERM)
    logger.info("kill servers")


def test_consistency_of_all_batch_sovler(server):

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        create_default_logger(td_path, "test_datagen")

        for n_problem in [1, 8]:  # to test edge case
            n_problem_inner = 2

            task = TabletopBoxRightArmReachingTask.sample(1, standard=True)
            solcon = OMPLSolverConfig(10000, n_max_satisfaction_trial=100)
            solver = OMPLSolver.setup(task.export_problems()[0], solcon)
            result = solver.solve()
            assert result.traj is not None

            init_solutions = [result.traj] * n_problem
            # set standard = True for testing purpose
            tasks = [
                TabletopBoxRightArmReachingTask.sample(n_problem_inner) for _ in range(n_problem)
            ]
            batch_solver_list: List[BatchProblemSolver] = []
            mp_batch_solver = MultiProcessBatchProblemSolver[
                TabletopBoxRightArmReachingTask, OMPLSolverConfig, OMPLSolverResult
            ](OMPLSolver, solcon, 2)
            assert mp_batch_solver.n_process == 2
            batch_solver_list.append(mp_batch_solver)

            specs = (ServerSpec("localhost", 8081, 1.0), ServerSpec("localhost", 8082, 1.0))
            batch_solver = DistributedBatchProblemSolver[
                TabletopBoxRightArmReachingTask, OMPLSolverConfig, OMPLSolverResult
            ](OMPLSolver, solcon, specs, n_measure_sample=1)
            batch_solver_list.append(batch_solver)

            # compare generated nit and success
            # we wanted to directly compare results_list but somehow, pickling-unpickling process
            # change the hash value. so...
            nits_list = []
            successes_list = []
            for batch_solver in batch_solver_list:  # type: ignore
                print(batch_solver)
                results_list = batch_solver.solve_batch(tasks, init_solutions)
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

            # NOTE: it seems that osqp solve results sometimes slightly different though
            # the same problem is provided.. maybe random variable is used inside???
            assert len(set(nits_list)) == 1
            assert len(set(successes_list)) == 1


def test_consistency_of_all_batch_sampler(server):
    specs = (ServerSpec("localhost", 8081, 1.0), ServerSpec("localhost", 8082, 1.0))

    sampler_list: List[BatchProblemSampler[TabletopBoxRightArmReachingTask]] = []
    sampler_list.append(MultiProcessBatchProblemSampler(1))
    sampler_list.append(DistributeBatchProblemSampler[TabletopBoxRightArmReachingTask](specs))

    n_problem_inner = 5
    pool_list: List[PredicatedProblemPool] = []
    pool_base = TrivialProblemPool(TabletopBoxRightArmReachingTask, n_problem_inner)
    pool_list.append(pool_base.as_predicated())
    pool_list.append(pool_base.make_predicated(SimplePredicate(), 40))

    for n_sample in [1, 2, 20]:  # to test edge case
        for pool in pool_list:
            for sampler in sampler_list:
                samples = sampler.sample_batch(n_sample, pool)
                assert len(samples) == n_sample
                assert len(samples[0].descriptions) == n_problem_inner


def test_create_dataset():
    n_task = 4
    n_problem_inner = 2

    task = TabletopBoxRightArmReachingTask.sample(1, standard=True)
    solcon = OMPLSolverConfig(10000, n_max_satisfaction_trial=100)
    solver = OMPLSolver.setup(task.export_problems()[0], solcon)
    result = solver.solve()
    assert result.traj is not None

    init_solutions = [result.traj] * n_task

    problems = [TabletopBoxRightArmReachingTask.sample(n_problem_inner) for _ in range(n_task)]

    batch_solver = MultiProcessBatchProblemSolver[
        TabletopBoxRightArmReachingTask, OMPLSolverConfig, OMPLSolverResult
    ](OMPLSolver, solcon, 2)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        batch_solver.create_dataset(problems, init_solutions, td_path, n_process=None)

        dataset = LazyDecomplessDataset.load(td_path, RawData)
        assert len(dataset) == n_task
        for i in range(n_task):
            data = dataset.get_data(np.array([i]))[0]
            assert len(data.results) == n_problem_inner
        loader = LazyDecomplessDataLoader(dataset, batch_size=1)
        for sample in loader:
            pass
