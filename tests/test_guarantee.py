import tempfile
from pathlib import Path

import numpy as np
import torch
from mohou.trainer import TrainConfig

from hifuku.guarantee.algorithm import (
    LibrarySamplerConfig,
    MultiProcessDatasetGenerator,
    SimpleFixedProblemPool,
    SolutionLibrarySampler,
)
from hifuku.neuralnet import VoxelAutoEncoder, VoxelAutoEncoderConfig
from hifuku.threedim.tabletop import TabletopPlanningProblem
from hifuku.types import RawData


def test_MultiProcessDatasetGenerator():
    gen = MultiProcessDatasetGenerator(TabletopPlanningProblem, n_process=2)
    prob_stan = TabletopPlanningProblem.create_standard()
    sol = prob_stan.solve()[0]
    assert sol.success
    n_problem = 4
    n_problem_inner = 10

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        gen.generate(sol.x, n_problem, n_problem_inner, td_path)

        file_path_list = list(td_path.iterdir())
        assert len(file_path_list) == n_problem

        for file_path in file_path_list:
            rawdata = RawData.load(file_path, decompress=True)
            assert len(rawdata.descriptions) == n_problem_inner


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
            lib_sampler = SolutionLibrarySampler.initialize(
                problem_type, ae_model, gen, lconfig, validation_pool
            )
            # init
            lib_sampler.step_active_sampling(td_path)
            # active sampling
            lib_sampler.step_active_sampling(td_path)


if __name__ == "__main__":
    test_SolutionLibrarySampler()
