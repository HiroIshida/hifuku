import os
import warnings

import numpy as np
from llazy.generation import DataGenerationTask, DataGenerationTaskArg
from mohou.file import get_project_path

from hifuku.threedim.tabletop import TableTopWorld
from hifuku.types import RawMeshData

warnings.filterwarnings("ignore", message="Values in x were outside bounds during")
warnings.filterwarnings("ignore", message="texture specified in URDF is not supported")


def split_number(num, div):
    return [num // div + (1 if x < num % div else 0) for x in range(div)]


class TabletopIKGenerationTask(DataGenerationTask[RawMeshData]):
    def post_init_hook(self) -> None:
        pass

    def generate_single_data(self) -> RawMeshData:
        world = TableTopWorld.sample()
        gridsdf = world.compute_exact_gridsdf(fill_value=2.0)
        gridsdf = gridsdf.get_quantized()
        mesh = gridsdf.values.reshape(*gridsdf.grid.sizes)
        return RawMeshData(mesh)


if __name__ == "__main__":
    # n_problem = 300000
    n_problem = 20000

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

    if n_process > 1:
        process_list = []
        for idx_process, number in enumerate(numbers):
            show_process_bar = idx_process == 1
            arg = DataGenerationTaskArg(number, show_process_bar, chunk_dir_path, extension=".npz")
            p = TabletopIKGenerationTask(arg)
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()
    else:
        arg = DataGenerationTaskArg(n_problem, True, chunk_dir_path, extension=".npz")
        task = TabletopIKGenerationTask(arg)
        task.run()
