import os
import warnings

import numpy as np
from mohou.file import get_project_path
from skplan.solver.optimization import OsqpSqpPlanner

from hifuku.llazy.generation import DataGenerationTask, DataGenerationTaskArg
from hifuku.threedim.tabletop import TabletopPlanningProblem
from hifuku.types import RawData

warnings.filterwarnings("ignore", message="Values in x were outside bounds during")
warnings.filterwarnings("ignore", message="texture specified in URDF is not supported")


def split_number(num, div):
    return [num // div + (1 if x < num % div else 0) for x in range(div)]


class TabletopIKGenerationTask(DataGenerationTask[RawData]):
    def post_init_hook(self) -> None:
        pass

    def generate_single_data(self) -> RawData:
        assert self.arg.info is not None
        x_init = self.arg.info["init_solution"]
        config = OsqpSqpPlanner.SolverConfig(verbose=False)
        problem = TabletopPlanningProblem.sample(n_pose=50)
        results = problem.solve(x_init, config=config)
        print([r.success for r in results])
        data = RawData.create(problem, results, x_init, config)
        return data


if __name__ == "__main__":
    # n_problem = 300000
    n_problem = 1000

    av_init = np.zeros(10)

    pp = get_project_path("tabletop_ik")
    pp.mkdir(exist_ok=True)

    chunk_dir_path = pp / "chunk"
    chunk_dir_path.mkdir(exist_ok=True)

    cpu_num = os.cpu_count()
    assert cpu_num is not None
    n_process = int(cpu_num * 0.5)
    # n_process = 1
    assert n_process is not None
    numbers = split_number(n_problem, n_process)

    # create initial solution
    np.random.seed(0)
    problem = TabletopPlanningProblem.sample(n_pose=1)
    result = problem.solve()[0]
    assert result.success
    x_init = result.x.flatten()
    print(x_init)
    info = {"init_solution": x_init}

    if n_process > 1:
        process_list = []
        for idx_process, number in enumerate(numbers):
            show_process_bar = idx_process == 1
            core = [idx_process * 2, idx_process * 2 + 1]
            arg = DataGenerationTaskArg(
                number, show_process_bar, chunk_dir_path, extension=".npz", info=info, cpu_core=core
            )
            p = TabletopIKGenerationTask(arg)
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()
    else:
        arg = DataGenerationTaskArg(
            n_problem, True, chunk_dir_path, extension=".npz", info=info, cpu_core=[0, 1]
        )
        task = TabletopIKGenerationTask(arg)
        task.run()
