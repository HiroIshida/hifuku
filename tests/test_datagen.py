import tempfile
from pathlib import Path

from hifuku.datagen import MultiProcessDatasetGenerator
from hifuku.threedim.tabletop import TabletopPlanningProblem
from hifuku.types import RawData


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
