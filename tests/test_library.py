import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from mohou.trainer import TrainConfig
from skmp.solver.nlp_solver import (
    SQPBasedSolver,
    SQPBasedSolverConfig,
    SQPBasedSolverResult,
)
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skmp.trajectory import Trajectory

from hifuku.datagen import (
    MultiProcessBatchProblemSampler,
    MultiProcessBatchProblemSolver,
)
from hifuku.library import (
    LibrarySamplerConfig,
    SimpleSolutionLibrarySampler,
    SolutionLibrary,
)
from hifuku.neuralnet import VoxelAutoEncoder, VoxelAutoEncoderConfig
from hifuku.rpbench_wrap import TabletopBoxRightArmReachingTask
from hifuku.utils import create_default_logger


def _test_compute_real_itervals():
    # compute default solution
    standard_problem = TabletopBoxRightArmReachingTask.sample(1, True).export_problems()[0]
    init_solution: Optional[Trajectory] = None
    solcon = OMPLSolverConfig(n_max_call=3000, n_max_satisfaction_trial=100)
    solver = OMPLSolver.init(solcon)
    for _ in range(10):
        solver.setup(standard_problem)
        res = solver.solve()
        if res.traj is not None:
            init_solution = res.traj
            break
    assert init_solution is not None

    n_problem = 10
    problems = [TabletopBoxRightArmReachingTask.sample(1) for _ in range(n_problem)]
    init_solutions = [init_solution] * n_problem

    nlp_solcon = SQPBasedSolverConfig(n_wp=15, n_max_call=10)
    solver_mp = MultiProcessBatchProblemSolver[SQPBasedSolverConfig, SQPBasedSolverResult](
        SQPBasedSolver, nlp_solcon, n_process=4
    )
    solver_sp = MultiProcessBatchProblemSolver[SQPBasedSolverConfig, SQPBasedSolverResult](
        SQPBasedSolver, nlp_solcon, n_process=1
    )
    results_mp = solver_mp.solve_batch(problems, init_solutions)
    results_sp = solver_sp.solve_batch(problems, init_solutions)
    assert len(results_mp) == n_problem
    assert len(results_sp) == n_problem

    itervals_mp = [r[0].n_call for r in results_mp]
    itervals_sp = [r[0].n_call for r in results_sp]
    # NOTE: these itervals are supposed to match when nlp solver is deterministic.
    # However, osqp solver is actualy has non-deterministic part inside, thus
    # sometime results dont match.
    # see: https://osqp.discourse.group/t/what-settings-to-choose-to-ensure-reproducibility/64
    n_mismatch = 0
    for iterval_sp, interval_mp in zip(itervals_sp, itervals_mp):
        if itervals_sp != itervals_mp:
            n_mismatch += 1
    assert n_mismatch < 3


def test_SolutionLibrarySampler():
    problem_type = TabletopBoxRightArmReachingTask
    nlp_solcon = SQPBasedSolverConfig(
        n_wp=15, n_max_call=10, motion_step_satisfaction="debug_ignore"
    )
    solver = MultiProcessBatchProblemSolver[SQPBasedSolverConfig, SQPBasedSolverResult](
        SQPBasedSolver, nlp_solcon, n_process=2
    )
    sampler = MultiProcessBatchProblemSampler[TabletopBoxRightArmReachingTask](n_process=2)
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
        ae_model.loss_called = True  # mock that model is already trained
        ae_model.put_on_device(device)
        pool_validation = [problem_type.sample(1) for _ in range(10)]

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            create_default_logger(td_path, "test_trajectorylib")
            lib_sampler = SimpleSolutionLibrarySampler.initialize(
                problem_type,
                SQPBasedSolver,
                nlp_solcon,
                ae_model,
                lconfig,
                problems_validation=pool_validation,
                solver=solver,
                sampler=sampler,
            )
            # init
            lib_sampler.step_active_sampling(td_path)
            # active sampling
            lib_sampler.step_active_sampling(td_path)

            # test load
            lib_load = SolutionLibrary.load(td_path, problem_type, SQPBasedSolver)[0]

            # compare
            for _ in range(10):
                problem = problem_type.sample(1)
                iters = lib_sampler.library._infer_iteration_num(problem)
                iters_again = lib_load._infer_iteration_num(problem)
                np.testing.assert_almost_equal(iters, iters_again)


if __name__ == "__main__":
    test_SolutionLibrarySampler()
