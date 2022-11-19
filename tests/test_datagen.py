import tempfile
from pathlib import Path

from skplan.solver.constraint import ConstraintSatisfactionFail

from hifuku.datagen import MultiProcessDatasetGenerator
from hifuku.threedim.tabletop import TabletopPlanningProblem
from hifuku.types import RawData


def test_MultiProcessDatasetGenerator():
    gen = MultiProcessDatasetGenerator(TabletopPlanningProblem, n_process=2)
    prob_stan = TabletopPlanningProblem.create_standard()

    while True:
        try:
            sol = prob_stan.solve()[0]
            if sol.success:
                break
        except ConstraintSatisfactionFail:
            continue
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
