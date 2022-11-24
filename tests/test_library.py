import tempfile
from pathlib import Path

import numpy as np
import torch
from mohou.trainer import TrainConfig

from hifuku.datagen import MultiProcessDatasetGenerator
from hifuku.library import (
    LibrarySamplerConfig,
    MultiProcessProblemSolver,
    SimpleFixedProblemPool,
    SolutionLibrary,
    SolutionLibrarySampler,
)
from hifuku.neuralnet import VoxelAutoEncoder, VoxelAutoEncoderConfig
from hifuku.threedim.tabletop import TabletopPlanningProblem
from hifuku.utils import create_default_logger


def test_compute_real_itervals():
    x_init = TabletopPlanningProblem.get_default_init_solution()
    n_problem = 10
    problems = [TabletopPlanningProblem.sample(1) for _ in range(n_problem)]
    init_solutions = [x_init] * n_problem
    results_mp = MultiProcessProblemSolver.solve(problems, init_solutions, n_process=4)
    results_sp = MultiProcessProblemSolver.solve(problems, init_solutions, n_process=1)
    assert len(results_mp) == n_problem
    assert len(results_sp) == n_problem

    itervals_mp = [r.nit for r in results_mp]
    itervals_sp = [r.nit for r in results_sp]
    assert itervals_mp == itervals_sp


def test_SolutionLibrarySampler():
    problem_type = TabletopPlanningProblem
    gen = MultiProcessDatasetGenerator(problem_type, n_process=2)
    tconfig = TrainConfig(n_epoch=1)
    lconfig = LibrarySamplerConfig(
        n_problem=10,
        n_problem_inner=1,
        train_config=tconfig,
        n_solution_candidate=2,
        n_difficult_problem=5,
        solvable_threshold_factor=0.0,
        difficult_threshold_factor=-np.inf,  # all pass
        acceptable_false_positive_rate=1.0,
    )  # all pass

    test_devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        test_devices.append(torch.device("cuda"))

    for device in test_devices:
        ae_model = VoxelAutoEncoder(VoxelAutoEncoderConfig())
        ae_model.put_on_device(device)
        validation_pool = SimpleFixedProblemPool.initialize(problem_type, 10)

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            create_default_logger(td_path, "test_trajectorylib")
            lib_sampler = SolutionLibrarySampler.initialize(
                problem_type, ae_model, gen, lconfig, validation_pool
            )
            # init
            lib_sampler.step_active_sampling(td_path)
            # active sampling
            lib_sampler.step_active_sampling(td_path)

            # test load
            lib_load = SolutionLibrary.load(td_path, problem_type)[0]

            # compare
            for _ in range(10):
                problem = problem_type.sample(1)
                iters = lib_sampler.library._infer_iteration_num(problem)
                iters_again = lib_load._infer_iteration_num(problem)
                np.testing.assert_almost_equal(iters, iters_again)


if __name__ == "__main__":
    test_SolutionLibrarySampler()
