import tempfile
from pathlib import Path

import numpy as np

from hifuku.datagen import (
    DataGenerationTaskArg,
    HifukuDataGenerationTask,
    MultiProcessDatasetGenerator,
)
from hifuku.llazy.dataset import LazyDecomplessDataset
from hifuku.threedim.tabletop import TabletopPlanningProblem
from hifuku.types import PredicateInterface, RawData


class SimplePredicate(PredicateInterface[TabletopPlanningProblem]):
    def __call__(self, problem: TabletopPlanningProblem) -> bool:
        assert problem.n_problem() == 1
        pose = problem.target_pose_list[0]
        is_y_positive = pose.worldpos()[1] > 0.0
        return is_y_positive


def test_DataGenerationTask():

    with tempfile.TemporaryDirectory() as td:
        x_init = TabletopPlanningProblem.get_default_init_solution()
        td_path = Path(td)
        arg = DataGenerationTaskArg(4, False, td_path, extension=".npz")
        task = HifukuDataGenerationTask(
            arg, TabletopPlanningProblem, 5, x_init, predicate=SimplePredicate()
        )
        task.run()

        # load and check
        dataset = LazyDecomplessDataset.load(td_path, RawData, n_worker=1)
        data_list = dataset.get_data(np.array([0, 1, 2, 3]))

        for data in data_list:
            desc = data.descriptions[0]
            assert len(desc) == 12  # assume pose of target + pose of table is concatted
            is_y_positive = desc[1] > 0.0
            assert is_y_positive


def test_MultiProcessDatasetGenerator():
    gen = MultiProcessDatasetGenerator(TabletopPlanningProblem, n_process=2)
    init_solution = TabletopPlanningProblem.get_default_init_solution()
    n_problem = 4
    n_problem_inner = 10

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        gen.generate(init_solution, n_problem, n_problem_inner, td_path)

        file_path_list = list(td_path.iterdir())
        assert len(file_path_list) == n_problem

        for file_path in file_path_list:
            rawdata = RawData.load(file_path, decompress=True)
            assert len(rawdata.descriptions) == n_problem_inner
