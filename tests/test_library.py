import tempfile
import time
from pathlib import Path
from typing import Type

import torch
from mohou.trainer import TrainConfig

from hifuku.domain import DomainProtocol, DummyDomain, DummyMeshDomain
from hifuku.library import (
    ActiveSamplerHistory,
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
        n_validation=1000,
        n_validation_inner=1,
    )
    _CLAMP_FACTOR[0] = 1.5

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
            create_default_logger(td_path, "test_trajectorylib")

            args = (problem_type, solver_type, solcon, ae_model, lconfig, td_path)
            kwargs = {"solver": solver, "sampler": sampler, "device": device}

            # main test
            lib_sampler = SimpleSolutionLibrarySampler.initialize(*args, **kwargs)
            ts = time.time()
            lib_sampler.step_active_sampling()
            lib_sampler.step_active_sampling()
            elapsed = time.time() - ts
            assert (
                abs(lib_sampler.sampler_history.total_time - elapsed) < 1.0
            )  # sec. might depend on testing environment...
            coverage = lib_sampler.sampler_history.coverage_est_history[-1]
            # as dummy domain is easy, 0.3 should be satisfied (but might not in quite small probability)
            assert coverage > 0.3, f"coverage={coverage}"

            # test warm start
            lib = SolutionLibrary.load(td_path, problem_type, solver_type, device=device)[0]
            state = ActiveSamplerHistory.load(td_path)
            assert state.total_iter == 2
            lib_sampler = SimpleSolutionLibrarySampler.initialize(*args, **kwargs)
            lib_sampler.setup_warmstart(state, lib)
            time.sleep(2.0)  # for checking total_time
            assert len(lib_sampler.presampled_tasks_paramss) > 0, "presampled tasks are not loaded"
            ts = time.time()
            lib_sampler.step_active_sampling()
            elpased_warm = time.time() - ts + elapsed
            assert (
                abs(lib_sampler.sampler_history.total_time - elpased_warm) < 1.0
            )  # sec. might depend on testing environment...
            assert lib_sampler.sampler_history.total_iter == 3

            coverage_after_warm = lib_sampler.sampler_history.coverage_est_history[-1]
            assert coverage_after_warm > coverage, "presumably warm start did not work"
            # as dummy domain is easy, 0.5 should be satisfied (but might not in quite small probability)
            assert coverage_after_warm > 0.5, f"coverage_after_warm={coverage_after_warm}"

            # test coverage estimation's consistency
            # diff = (
            #     lib_sampler.library.measure_coverage(lib_sampler.problems_validation)
            #     - coverage_after_warm
            # )
            # assert abs(diff) < 1e-6, f"diff={diff}"


def test_SolutionLibrarySampler():
    _test_SolutionLibrarySampler(DummyDomain, False)
    _test_SolutionLibrarySampler(DummyMeshDomain, False)


if __name__ == "__main__":
    test_SolutionLibrarySampler()
