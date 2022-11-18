import tempfile
from pathlib import Path

from mohou.trainer import TrainConfig

from hifuku.guarantee.algorithm import (
    LibrarySamplerConfig,
    MultiProcessDatasetGenerator,
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
    gen = MultiProcessDatasetGenerator(TabletopPlanningProblem, n_process=2)
    tconfig = TrainConfig(n_epoch=1)
    difficult_threshold = 0.0  # all pass
    lconfig = LibrarySamplerConfig(10, 1, tconfig, 5, difficult_threshold)
    ae_model = VoxelAutoEncoder(VoxelAutoEncoderConfig())

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        lib_sampler = SolutionLibrarySampler.initialize(
            TabletopPlanningProblem, ae_model, gen, lconfig
        )
        # init
        lib_sampler.step_active_sampling(td_path)
        # active sampling
        lib_sampler.step_active_sampling(td_path)


if __name__ == "__main__":
    test_SolutionLibrarySampler()
