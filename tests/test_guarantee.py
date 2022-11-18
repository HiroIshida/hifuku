from hifuku.guarantee.algorithm import MultiProcessDatasetGenerator
from hifuku.threedim.tabletop import TabletopPlanningProblem
from hifuku.types import RawData


def test_MultiProcessDatasetGenerator():
    gen = MultiProcessDatasetGenerator(TabletopPlanningProblem, n_process=2)
    prob_stan = TabletopPlanningProblem.create_standard()
    sol = prob_stan.solve()[0]
    assert sol.success
    n_problem = 4
    n_problem_inner = 10
    dataset_path = gen.generate(sol.x, n_problem, n_problem_inner)

    file_path_list = list(dataset_path.iterdir())
    assert len(file_path_list) == n_problem

    for file_path in file_path_list:
        rawdata = RawData.load(file_path, decompress=True)
        assert len(rawdata.descriptions) == n_problem_inner


if __name__ == "__main__":
    test_MultiProcessDatasetGenerator()
