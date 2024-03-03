import tempfile
import time
from pathlib import Path
from typing import Type

import pytest
import torch
from mohou.trainer import TrainConfig

from hifuku.core import (
    ActiveSamplerHistory,
    LibraryBasedGuaranteedSolver,
    LibrarySamplerConfig,
    SimpleSolutionLibrarySampler,
    SolutionLibrary,
)
from hifuku.domain import DomainProtocol, DummyDomain, DummyMeshDomain
from hifuku.neuralnet import (
    AutoEncoderBase,
    AutoEncoderConfig,
    NullAutoEncoder,
    PixelAutoEncoder,
)
from hifuku.script_utils import create_default_logger


def _test_SolutionLibrarySampler(
    domain: Type[DomainProtocol], train_with_encoder: bool, device: torch.device
):
    task_type = domain.task_type
    solcon = domain.solver_config
    solver_type = domain.solver_type
    domain.auto_encoder_project_name
    batch_solver = domain.get_multiprocess_batch_solver(2)
    sampler = domain.get_multiprocess_batch_sampler(2)

    lconfig = LibrarySamplerConfig(
        acceptable_false_positive_rate=0.2,
        n_difficult=100,
        n_solution_candidate=10,
        sampling_number_factor=1000,
        train_config=TrainConfig(n_epoch=20, learning_rate=0.01),
        costpred_model_config={"layers": [64, 64, 64]},
        n_optimize_biases_batch=50,
        n_validation=1000,
        n_validation_inner=1,
        clamp_factor=1.5,
    )

    test_devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        test_devices.append(torch.device("cuda"))

    for device in test_devices:
        ae_model: AutoEncoderBase
        if domain.auto_encoder_type is NullAutoEncoder:
            ae_model = NullAutoEncoder()
            ae_model.put_on_device(device)
        else:
            ae_model = PixelAutoEncoder(AutoEncoderConfig(n_grid=56))
            ae_model.loss_called = True  # mock that model is already trained
            ae_model.put_on_device(device)

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            logger = create_default_logger(td_path, "test_trajectorylib")

            args = (task_type, solver_type, solcon, ae_model, lconfig, td_path)
            kwargs = {"solver": batch_solver, "sampler": sampler, "device": device}

            # main test
            lib_sampler = SimpleSolutionLibrarySampler.initialize(*args, **kwargs)
            elapsed = 0.0
            for it in range(2):
                ts = time.time()
                lib_sampler.step_active_sampling()
                elapsed += time.time() - ts
                lib_sampler.sampler_history.elapsed_time_history
                assert (
                    abs(lib_sampler.sampler_history.total_time - elapsed) < 1.0
                ), f"iter={it}, total_time={lib_sampler.sampler_history.total_time} vs elapsed={elapsed}. all history={lib_sampler.sampler_history.elapsed_time_history}"

                # test coverage estimation's consistency
                a = lib_sampler.sampler_history.coverage_est_history[-1]
                b = lib_sampler.measure_coverage(lib_sampler.tasks_validation)
                assert (
                    abs(a - b) < 1e-6
                ), f"iter={it} coverage estimation is not consistent: {a} vs {b}"

            coverage = lib_sampler.sampler_history.coverage_est_history[-1]
            # as dummy domain is easy, 0.3 should be satisfied (but might not in quite small probability)
            assert coverage > 0.3, f"iter={it}, coverage={coverage}"

            test_warmstart = True
            if test_warmstart:
                # test warm start
                lib = SolutionLibrary.load(td_path, device)
                state = ActiveSamplerHistory.load(td_path)
                assert state.total_iter == 2
                lib_sampler = SimpleSolutionLibrarySampler.initialize(*args, **kwargs)
                lib_sampler.setup_warmstart(state, lib)
                time.sleep(2.0)  # for checking total_time
                assert (
                    len(lib_sampler.presampled_tasks_params) > 0
                ), "presampled tasks are not loaded"
                ts = time.time()
                lib_sampler.step_active_sampling()
                elapsed += time.time() - ts
                assert (
                    abs(lib_sampler.sampler_history.total_time - elapsed) < 1.0
                )  # sec. might depend on testing environment...
                assert lib_sampler.sampler_history.total_iter == 3

                coverage_after_warm = lib_sampler.sampler_history.coverage_est_history[-1]
                # as dummy domain is easy, 0.5 should be satisfied (but might not in quite small probability)
                assert coverage_after_warm > 0.5, f"coverage_after_warm={coverage_after_warm}"

                # test coverage estimation's consistency
                a = lib_sampler.sampler_history.coverage_est_history[-1]
                b = lib_sampler.measure_coverage(lib_sampler.tasks_validation)
                assert abs(a - b) < 1e-6, f"coverage estimation is not consistent: {a} vs {b}"

            # test library based solver usage
            lib = lib_sampler.library
            solver = LibraryBasedGuaranteedSolver.init(lib, solver_type, solcon)
            est_true_count = 0
            fp_count = 0
            n = 300
            for _ in range(n):
                task = task_type.sample()
                solver.setup(task)
                ret = solver.solve()
                if solver.previous_est_positive:
                    est_true_count += 1
                    if ret.traj is None:
                        fp_count += 1
            coverage = est_true_count / n
            coverage_expected = lib_sampler.sampler_history.coverage_est_history[-1]
            assert abs(coverage - lib_sampler.sampler_history.coverage_est_history[-1]) < 0.2
            logger.info(f"coverage={coverage}, coverage_expected={coverage_expected}")
            fprate = fp_count / est_true_count
            fprate_expected = lconfig.acceptable_false_positive_rate
            logger.info(f"fprate={fprate}, fprate_expected={fprate_expected}")
            assert abs(fprate - lconfig.acceptable_false_positive_rate) < 0.1
            logger.info(f"fprate={fprate}, fprate_expected={fprate_expected}")


dom = [DummyDomain, DummyMeshDomain]
encode = [False, True]
if torch.cuda.is_available():
    device = [torch.device("cpu"), torch.device("cuda")]
else:
    device = [torch.device("cpu")]  # in ci
parameter_matrix = [(d, e, dev) for d in dom for e in encode for dev in device]


@pytest.mark.parametrize("domain, train_with_encoder, device", parameter_matrix)
def test_SolutionLibrarySampler(domain, train_with_encoder, device):
    _test_SolutionLibrarySampler(domain, train_with_encoder, device)


if __name__ == "__main__":
    _test_SolutionLibrarySampler(DummyDomain, False, torch.device("cpu"))
