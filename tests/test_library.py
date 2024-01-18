import tempfile
from pathlib import Path
from typing import Optional, Type

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

from hifuku.datagen import MultiProcessBatchProblemSolver
from hifuku.domain import (
    DomainProtocol,
    DoubleIntegratorBubblySimple_SQP,
    TabletopOvenRightArmReachingTask,
)
from hifuku.library import (
    LibrarySamplerConfig,
    SimpleSolutionLibrarySampler,
    SolutionLibrary,
)
from hifuku.neuralnet import (
    AutoEncoderBase,
    AutoEncoderConfig,
    NullAutoEncoder,
    PixelAutoEncoder,
)
from hifuku.script_utils import create_default_logger


def _test_compute_real_itervals():
    # compute default solution
    standard_problem = TabletopOvenRightArmReachingTask.sample(1, True).export_problems()[0]
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
    problems = [TabletopOvenRightArmReachingTask.sample(1) for _ in range(n_problem)]
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


def _test_SolutionLibrarySampler(domain: Type[DomainProtocol], train_with_encoder: bool):
    problem_type = domain.task_type
    solcon = domain.solver_config
    solver_type = domain.solver_type
    aepp = domain.auto_encoder_project_name
    solver = domain.get_multiprocess_batch_solver(2)
    sampler = domain.get_multiprocess_batch_sampler(2)

    tconfig = TrainConfig(n_epoch=1)
    lconfig = LibrarySamplerConfig(
        n_problem_init=10,
        n_problem_mult_factor=1.2,
        n_problem_max=13,
        n_problem_inner=1,
        train_config=tconfig,
        n_solution_candidate=2,
        n_difficult_problem=5,
        solvable_threshold_factor=0.0,
        difficult_threshold_factor=-np.inf,  # all pass
        acceptable_false_positive_rate=1.0,
        sample_from_difficult_region=False,
        ignore_useless_traj=False,
        train_with_encoder=train_with_encoder,
        n_validation=50,
        n_validation_inner=2,
        n_determine_batch=10,
        candidate_sample_scale=5,
    )  # all pass

    test_devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        test_devices.append(torch.device("cuda"))

    for device in test_devices:
        ae_model: AutoEncoderBase
        if aepp is None:
            ae_model = NullAutoEncoder()
        else:
            ae_model = PixelAutoEncoder(AutoEncoderConfig())
            ae_model.loss_called = True  # mock that model is already trained
            ae_model.put_on_device(device)
        pool_validation = [problem_type.sample(1) for _ in range(10)]

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            create_default_logger(td_path, "test_trajectorylib")
            lib_sampler = SimpleSolutionLibrarySampler.initialize(
                problem_type,
                solver_type,
                solcon,
                ae_model,
                lconfig,
                td_path,
                problems_validation=pool_validation,
                solver=solver,
                sampler=sampler,
                adjust_margins=False,
            )
            # init k=0
            lib_sampler.step_active_sampling()
            assert lib_sampler.library._n_problem_now == 10
            # active sampling k=1
            lib_sampler.step_active_sampling()
            assert lib_sampler.library._n_problem_now == 12

            # active sampling k=2
            lib_sampler.step_active_sampling()
            assert lib_sampler.library._n_problem_now == 13  # note n_problem_max

            # test load
            lib_load = SolutionLibrary.load(td_path, problem_type, SQPBasedSolver)[0]

            # compare
            for _ in range(10):
                problem = problem_type.sample(1)
                iters = lib_sampler.library._infer_iteration_num(problem)
                iters_again = lib_load._infer_iteration_num(problem)
                np.testing.assert_almost_equal(iters, iters_again)


def test_SolutionLibrarySampler():
    _test_SolutionLibrarySampler(DoubleIntegratorBubblySimple_SQP, False)
    _test_SolutionLibrarySampler(DoubleIntegratorBubblySimple_SQP, True)


if __name__ == "__main__":
    _test_SolutionLibrarySampler(DoubleIntegratorBubblySimple_SQP, True)
