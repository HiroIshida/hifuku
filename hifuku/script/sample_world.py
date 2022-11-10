import os
import warnings

import numpy as np
from llazy.generation import DataGenerationTask, DataGenerationTaskArg
from mohou.file import get_project_path
from skplan.solver.optimization import IKConfig

from hifuku.threedim.tabletop import TabletopIKProblem
from hifuku.types import RawData

warnings.filterwarnings("ignore", message="Values in x were outside bounds during")
warnings.filterwarnings("ignore", message="texture specified in URDF is not supported")


def split_number(num, div):
    return [num // div + (1 if x < num % div else 0) for x in range(div)]


class TabletopIKGenerationTask(DataGenerationTask[RawData]):
    def post_init_hook(self) -> None:
        pass

    def generate_single_data(self) -> RawData:
        ik_config = IKConfig(disp=False)
        problem = TabletopIKProblem.sample(n_pose=20)
        results = problem.solve(av_init, config=ik_config)
        data = RawData.create(problem, results)
        return data


if __name__ == "__main__":
    # n_problem = 300000
    n_problem = 10000

    av_init = np.zeros(10)

    pp = get_project_path("tabletop_ik")
    pp.mkdir(exist_ok=True)

    chunk_dir_path = pp / "chunk"
    chunk_dir_path.mkdir(exist_ok=True)

    cpu_num = os.cpu_count()
    assert cpu_num is not None
    n_process = int(cpu_num * 0.5)
    assert n_process is not None
    numbers = split_number(n_problem, n_process)

    process_list = []
    for idx_process, number in enumerate(numbers):
        show_process_bar = idx_process == 1
        arg = DataGenerationTaskArg(
            idx_process, number, show_process_bar, chunk_dir_path, extension=".npz"
        )
        p = TabletopIKGenerationTask(arg)
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()
