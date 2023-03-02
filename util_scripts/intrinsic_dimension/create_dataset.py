import argparse
import pickle
import uuid
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Type

import numpy as np
import threadpoolctl
import tqdm
from rpbench.interface import TaskBase

from hifuku.datagen.utils import split_number
from hifuku.domain import DomainProvider
from hifuku.library.core import SolutionLibrary
from hifuku.script_utils import DomainSelector, load_library
from hifuku.utils import get_random_seed, num_torch_thread


@dataclass
class Sampler:
    lib: SolutionLibrary
    task_type: Type[TaskBase]
    n_sample: int
    dump_path: Path
    show_progress_bar: bool = False

    def run(self):
        get_random_seed()
        random_seed = get_random_seed()
        print("random seed set to {}".format(random_seed))
        np.random.seed(random_seed)
        disable_tqdm = not self.show_progress_bar

        n_saved = 0
        threashold = self.lib.success_iter_threshold()
        with tqdm.tqdm(total=self.n_sample, disable=disable_tqdm) as pbar:
            with num_torch_thread(1):
                with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
                    while n_saved < self.n_sample:
                        task1 = self.task_type.sample(1)
                        iter_nums = self.lib._infer_iteration_num(task1).flatten()
                        bools_feasible1 = iter_nums < threashold

                        task2 = self.task_type.sample(1)
                        iter_nums = self.lib._infer_iteration_num(task1).flatten()
                        bools_feasible2 = iter_nums < threashold

                        bools_close = np.logical_and(bools_feasible1, bools_feasible2)
                        if np.any(bools_close):
                            file_path = self.dump_path / "{}.pkl".format(uuid.uuid4())
                            with file_path.open(mode="wb") as f:
                                pickle.dump((task1, task2), f)
                            n_saved += 1
                            pbar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-domain", type=str, default="tbrr_rrt", help="")
    args = parser.parse_args()
    domain_name: str = args.domain

    lib = load_library(
        domain_name,
        "cpu",
        limit_thread=True,
        project_path=Path(
            "~/.mohou/tabletop_solution_library-TBRR_RRT-n10000-alpha0.5"
        ).expanduser(),
    )

    domain: DomainProvider = DomainSelector[domain_name].value
    task_type: Type[TaskBase] = domain.get_task_type()

    n_sample = 10000
    n_process = 4

    n_alloc_list = split_number(n_sample, n_process)

    pp = Path("./tmp_result")
    pp.mkdir(exist_ok=True)

    process_list = []
    for i in range(n_process):
        sampler = Sampler(lib, task_type, n_alloc_list[i], pp, i == 0)
        p = Process(target=sampler.run, args=())
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()
