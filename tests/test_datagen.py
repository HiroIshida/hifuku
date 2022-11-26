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

from hifuku.datagen import (
    BatchProblemSampler,
    BatchProblemSolver,
    DistributeBatchProblemSampler,
    DistributedBatchProblemSolver,
    MultiProcessBatchProblemSampler,
    MultiProcessBatchProblemSolver,
)
from hifuku.llazy.dataset import LazyDecomplessDataLoader, LazyDecomplessDataset
from hifuku.pool import PredicatedIteratorProblemPool, SimpleProblemPool
from hifuku.testing_asset import SimplePredicate
from hifuku.threedim.tabletop import TabletopPlanningProblem
from hifuku.types import RawData

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def server():
    p1 = subprocess.Popen(
        "python3 -m hifuku.http_datagen.server -port 8081", shell=True, preexec_fn=os.setsid
    )
    p2 = subprocess.Popen(
        "python3 -m hifuku.http_datagen.server -port 8082", shell=True, preexec_fn=os.setsid
    )
    time.sleep(2)
    yield
    # https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true/4791612#4791612
    os.killpg(os.getpgid(p1.pid), signal.SIGTERM)
    os.killpg(os.getpgid(p2.pid), signal.SIGTERM)
    logger.info("kill servers")


def test_consistency_of_all_batch_sovler(server):
    for n_problem in [1, 8]:  # to test edge case
        n_problem_inner = 2
        init_solutions = [TabletopPlanningProblem.get_default_init_solution()] * n_problem
        problems = [TabletopPlanningProblem.sample(n_problem_inner) for _ in range(n_problem)]
        gen_list: List[BatchProblemSolver] = []
        gen_list.append(MultiProcessBatchProblemSolver(1))
        gen_list.append(MultiProcessBatchProblemSolver(2))

        hostport_pairs = [("localhost", 8081), ("localhost", 8082)]
        gen = DistributedBatchProblemSolver[TabletopPlanningProblem](
            hostport_pairs, n_measure_sample=1
        )
        gen_list.append(gen)

        # compare generated nit and success
        # we wanted to directly compare results_list but somehow, pickling-unpickling process
        # change the hash value. so...
        nits_list = []
        successes_list = []
        for gen in gen_list:  # type: ignore
            results_list = gen.solve_batch(problems, init_solutions)
            assert isinstance(results_list, list)
            assert len(results_list) == n_problem
            assert isinstance(results_list[0], tuple)
            assert len(results_list[0]) == n_problem_inner

            nits = []
            successes = []
            for results in results_list:
                nits.extend([r.nit for r in results])
                successes.extend([r.success for r in results])
            nits_list.append(tuple(nits))
            successes_list.append(tuple(successes))

        # NOTE: it seems that osqp solve results sometimes slightly different though
        # the same problem is provided.. maybe random variable is used inside???
        assert len(set(nits_list)) == 1
        assert len(set(successes_list)) == 1


def test_consistency_of_all_batch_sampler(server):
    hostport_pairs = [("localhost", 8081), ("localhost", 8082)]

    sampler_list: List[BatchProblemSampler[TabletopPlanningProblem]] = []
    sampler_list.append(MultiProcessBatchProblemSampler(1))
    sampler_list.append(MultiProcessBatchProblemSampler(2))
    sampler_list.append(DistributeBatchProblemSampler[TabletopPlanningProblem](hostport_pairs))

    n_problem_inner = 5
    pool_list: List[PredicatedIteratorProblemPool] = []
    pool_base = SimpleProblemPool(TabletopPlanningProblem, n_problem_inner)
    pool_list.append(pool_base.as_predicated())
    pool_list.append(pool_base.make_predicated(SimplePredicate(), 40))

    for n_sample in [1, 2, 20]:  # to test edge case
        for pool in pool_list:
            for sampler in sampler_list:
                samples = sampler.sample_batch(n_sample, pool)
                assert len(samples) == n_sample
                assert samples[0].n_problem() == n_problem_inner


def test_create_dataset():
    n_problem = 4
    n_problem_inner = 2
    init_solutions = [TabletopPlanningProblem.get_default_init_solution()] * n_problem
    problems = [TabletopPlanningProblem.sample(n_problem_inner) for _ in range(n_problem)]

    solver = MultiProcessBatchProblemSolver[TabletopPlanningProblem](2)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        solver.create_dataset(problems, init_solutions, td_path, n_process=None)

        dataset = LazyDecomplessDataset.load(td_path, RawData)
        assert len(dataset) == n_problem
        for i in range(n_problem):
            data = dataset.get_data(np.array([i]))[0]
            assert len(data.descriptions) == n_problem_inner
        loader = LazyDecomplessDataLoader(dataset, batch_size=1)
        for sample in loader:
            pass
