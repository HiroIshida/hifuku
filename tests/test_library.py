import tempfile
from pathlib import Path
from typing import Type

import torch
from mohou.trainer import TrainConfig

from hifuku.domain import DomainProtocol, DummyDomain, DummyMeshDomain
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
            lib_sampler = SimpleSolutionLibrarySampler.initialize(
                problem_type,
                solver_type,
                solcon,
                ae_model,
                lconfig,
                td_path,
                solver=solver,
                sampler=sampler,
                device=device,
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
