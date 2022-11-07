import uuid
import warnings
import zlib
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path

import dill
import numpy as np
import psutil
import tqdm
from mohou.file import get_project_path
from skplan.solver.optimization import IKConfig

from hifuku.tabletop import TabletopIKDataset, TabletopIKProblem

warnings.filterwarnings("ignore", message="Values in x were outside bounds during")
warnings.filterwarnings("ignore", message="texture specified in URDF is not supported")


# np.random.seed(1)


def split_number(num, div):
    return [num // div + (1 if x < num % div else 0) for x in range(div)]


@dataclass
class DataGenerationTaskArg:
    process_idx: int  # is not pid
    number: int
    show_process_bar: bool
    directory: Path


class DataGenerationTask(Process):
    arg: DataGenerationTaskArg

    def __init__(self, arg: DataGenerationTaskArg):
        self.arg = arg
        super().__init__()

    def run(self) -> None:
        self._run()

    def _run(self) -> bytes:
        np.random.seed(self.arg.process_idx)

        problems = []
        results = []

        ik_config = IKConfig(disp=False)
        disable_tqdm = not self.arg.show_process_bar
        for _ in tqdm.tqdm(range(self.arg.number), disable=disable_tqdm):
            problem = TabletopIKProblem.sample()
            result = problem.solve(av_init, config=ik_config)
            problems.append(problem)
            results.append(result)
        chunk = TabletopIKDataset(problems, results, av_init)
        compressed = zlib.compress(dill.dumps(chunk))

        file_path = chunk_dir_path / (str(uuid.uuid4()) + ".dill")
        with file_path.open(mode="wb") as f:
            f.write(compressed)
        return compressed


if __name__ == "__main__":
    n_problem = 300000

    av_init = np.zeros(10)

    p = get_project_path("tabletop_ik")
    p.mkdir(exist_ok=True)

    chunk_dir_path = p / "chunk"
    chunk_dir_path.mkdir(exist_ok=False)

    n_process = psutil.cpu_count(logical=False)
    numbers = split_number(n_problem, n_process)

    process_list = []
    for idx_process, number in enumerate(numbers):
        show_process_bar = idx_process == 1
        arg = DataGenerationTaskArg(idx_process, number, show_process_bar, chunk_dir_path)
        p = DataGenerationTask(arg)
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()
