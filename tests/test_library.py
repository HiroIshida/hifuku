import tempfile
from pathlib import Path
from typing import Optional, Type

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
    DummyDomain,
    DummyMeshDomain,
    TabletopOvenRightArmReachingTask,
)
from hifuku.library import (
    ActiveSamplerState,
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
from hifuku.types import _CLAMP_FACTOR


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
    domain.auto_encoder_project_name
    solver = domain.get_multiprocess_batch_solver(2)
    sampler = domain.get_multiprocess_batch_sampler(2)

    lconfig = LibrarySamplerConfig(
        n_difficult=100,
        n_solution_candidate=10,
        sampling_number_factor=1000,
        train_config=TrainConfig(n_epoch=20, learning_rate=0.01),
        iterpred_model_config={"layers": [64, 64, 64]},
        n_determine_batch=50,
    )
    _CLAMP_FACTOR[0] = 1.5

    test_devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        test_devices.append(torch.device("cuda"))

    for device in test_devices:
        ae_model: AutoEncoderBase
        if domain.auto_encoder_type is NullAutoEncoder:
            ae_model = NullAutoEncoder()
        else:
            ae_model = PixelAutoEncoder(AutoEncoderConfig(n_grid=56))
            ae_model.loss_called = True  # mock that model is already trained
            ae_model.put_on_device(device)

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
                solver=solver,
                sampler=sampler,
            )
            lib_sampler.step_active_sampling()
            lib_sampler.step_active_sampling()
            lib_sampler.step_active_sampling()
            assert lib_sampler.sampler_state.coverage_est_history[-1] > 0.5
            SolutionLibrary.load(td_path, problem_type, solver_type)[0]
            ActiveSamplerState.load(td_path)


def test_SolutionLibrarySampler():
    _test_SolutionLibrarySampler(DummyDomain, False)
    _test_SolutionLibrarySampler(DummyMeshDomain, False)


if __name__ == "__main__":
    test_SolutionLibrarySampler()
