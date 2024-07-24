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
from skmp.trajectory import Trajectory

from hifuku.coverage import RealEstAggregate
from hifuku.datagen import (
    BatchTaskSampler,
    BatchTaskSolver,
    DistributeBatchBiasesOptimizer,
    DistributeBatchTaskSampler,
    DistributedBatchTaskSolver,
    MultiProcesBatchBiasesOptimizer,
    MultiProcessBatchTaskSampler,
    MultiProcessBatchTaskSolver,
)
from hifuku.datagen.http_datagen.client import ServerSpec
from hifuku.domain import DoubleIntegratorBubblySimple_SQP
from hifuku.pool import PredicatedTaskPool, TaskPool
from hifuku.script_utils import create_default_logger

logger = logging.getLogger(__name__)

np.random.seed(0)
set_ompl_random_seed(0)
domain = DoubleIntegratorBubblySimple_SQP
task_type = domain.task_type


def kill_process_on_port(port):
    try:
        command = f"lsof -t -i:{port}"
        pid = subprocess.check_output(command, shell=True).decode().strip()

        if pid:
            subprocess.run(["kill", pid], check=True)
        else:
            print(f"No process found on port {port}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


@pytest.fixture(autouse=True, scope="session")
def server():
    # kill the process using 8081 and 8082
    kill_process_on_port(8081)
    kill_process_on_port(8082)

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
    for _ in range(20):
        task = task_type.sample()
        res = task.solve_default()
        if res.traj is not None:
            return res.traj
    assert False


def test_batch_solver_init_solutions():
    init_traj = compute_init_traj()
    mp_batch_solver = MultiProcessBatchTaskSolver(
        domain.solver_type, domain.solver_config, task_type, 2
    )
    n_task = 5
    task_params = np.array([task_type.sample().to_task_param() for _ in range(n_task)])
    mp_batch_solver.solve_batch(task_params, None)
    mp_batch_solver.solve_batch(task_params, [init_traj] * n_task)


def test_consistency_of_all_batch_sovler(server):
    init_traj = compute_init_traj()

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        create_default_logger(td_path, "test_datagen")

        for n_task in [1, 8]:  # to test edge case
            init_solutions = [init_traj] * n_task
            # set standard = True for testing purpose
            task_params = np.array([task_type.sample().to_task_param() for _ in range(n_task)])
            batch_solver_list: List[BatchTaskSolver] = []
            mp_batch_solver = MultiProcessBatchTaskSolver(
                domain.solver_type, domain.solver_config, task_type, n_process=2
            )
            assert mp_batch_solver.n_process == 2
            batch_solver_list.append(mp_batch_solver)

            specs = (ServerSpec("localhost", 8081, 1.0), ServerSpec("localhost", 8082, 1.0))
            batch_solver = DistributedBatchTaskSolver(
                domain.solver_type, domain.solver_config, task_type, specs, n_measure_sample=1
            )
            batch_solver_list.append(batch_solver)

            # compare generated nit and success
            # we wanted to directly compare results_list but somehow, pickling-unpickling process
            # change the hash value. so...
            nits_list = []
            successes_list = []
            for batch_solver in batch_solver_list:  # type: ignore
                print(batch_solver)
                results = batch_solver.solve_batch(
                    task_params, init_solutions, tmp_n_max_call_mult_factor=1.5
                )
                assert isinstance(results, list)
                assert len(results) == n_task

                nits = []
                successes = []
                for result in results:
                    nits.append(result.n_call)
                    successes.append(result.traj is not None)
                nits_list.append(tuple(nits))
                successes_list.append(tuple(successes))

                # check if multiplicatio by tmp_n_max_call_mult_factor is reset to the original value after the solve
                assert batch_solver.config.n_max_call == domain.solver_config.n_max_call

            # NOTE: it seems that osqp solve results sometimes slightly different though
            # the same task is provided.. maybe random variable is used inside???
            assert len(set(nits_list)) == 1
            assert len(set(successes_list)) == 1


def test_consistency_of_all_batch_sampler(server):
    specs = (ServerSpec("localhost", 8081, 1.0), ServerSpec("localhost", 8082, 1.0))

    sampler_list: List[BatchTaskSampler[task_type]] = []
    sampler_list.append(MultiProcessBatchTaskSampler(1))
    sampler_list.append(DistributeBatchTaskSampler[task_type](specs))

    pool_list: List[PredicatedTaskPool] = []
    pool_base = TaskPool(task_type)
    pool_list.append(pool_base.as_predicated())
    # pool_list.append(pool_base.make_predicated(SimplePredicate(), 40))

    for n_sample in [1, 2, 20]:  # to test edge case
        for pool in pool_list:
            for sampler in sampler_list:
                samples = sampler.sample_batch(n_sample, pool)
                assert samples.shape == (n_sample, task_type.get_task_dof())

                # in the parallel processing, the typical but difficult-to-find bug is
                # duplication of sample by forgetting to set peroper random seed.
                # check that no duplicate samples here.
                hash_vals = [hashlib.md5(pickle.dumps(s)).hexdigest() for s in samples]
                assert len(set(hash_vals)) == len(hash_vals)

                # Note: because different random seeds are used, the result of the sampling
                # is different. so we cannot compare the result of the sampling directly.


def test_batch_biases_optimizer(server):
    def f_real(x, c):
        return np.exp(-0.5 * ((x - c) ** 2)) + np.random.randn(len(x)) * 0.2

    def f_est(x, c):
        return np.exp(-0.5 * ((x - c) ** 2))

    x = np.random.rand(1000)
    real1 = f_real(x, 0.5)
    real2 = f_real(x, -0.5)
    est1 = f_est(x, 0.5)
    est2 = f_est(x, -0.5)
    cr1 = RealEstAggregate(real1, est1, 0.5)
    cr2 = RealEstAggregate(real2, est2, 0.5)

    specs = (ServerSpec("localhost", 8081, 1.0), ServerSpec("localhost", 8082, 1.0))
    optimizer_list = []
    optimizer_list.append(MultiProcesBatchBiasesOptimizer(4))
    optimizer_list.append(DistributeBatchBiasesOptimizer(specs))

    for optimizer in optimizer_list:
        optimizer.optimize_batch(8, [cr1, cr2], 5, 0.1, 5, None, None)
