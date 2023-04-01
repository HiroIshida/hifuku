import argparse
import pickle
import uuid
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Generic

import numpy as np
import threadpoolctl
import tqdm
from skmp.solver.interface import ConfigT, ResultT

from hifuku.datagen.utils import split_number
from hifuku.domain import DomainProvider
from hifuku.library.core import SolutionLibrary
from hifuku.pool import ProblemT
from hifuku.script_utils import DomainSelector, load_library
from hifuku.utils import get_random_seed, num_torch_thread


@dataclass
class Sampler(Generic[ProblemT, ConfigT, ResultT]):
    lib: SolutionLibrary[ProblemT, ConfigT, ResultT]
    domain: DomainProvider[ProblemT, ConfigT, ResultT]
    n_sample: int
    dump_path: Path
    show_progress_bar: bool = False

    def run(self):
        get_random_seed()
        random_seed = get_random_seed()
        print("random seed set to {}".format(random_seed))
        np.random.seed(random_seed)
        disable_tqdm = not self.show_progress_bar

        task_type = self.domain.get_task_type()
        solcon = self.domain.get_solver_config()
        solver_type = self.domain.get_solver_type()
        solver = solver_type.init(solcon)

        n_saved = 0
        threashold = self.lib.success_iter_threshold()
        with tqdm.tqdm(total=self.n_sample, disable=disable_tqdm) as pbar:
            with num_torch_thread(1):
                with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
                    while n_saved < self.n_sample:
                        task1 = task_type.sample(1)
                        iter_nums = self.lib._infer_iteration_num(task1).flatten()
                        bools_feasible1 = iter_nums < threashold

                        task2 = task_type.sample(1)
                        iter_nums = self.lib._infer_iteration_num(task1).flatten()
                        bools_feasible2 = iter_nums < threashold

                        bools_close = np.logical_and(bools_feasible1, bools_feasible2)
                        if np.any(bools_close):

                            problem1 = task1.export_problems()[0]
                            solver.setup(problem1)
                            res1 = solver.solve()
                            if res1.traj is None:
                                continue

                            problem2 = task2.export_problems()[0]
                            solver.setup(problem2)
                            res2_using_1 = solver.solve(res1.traj)
                            if res2_using_1.traj is None:
                                continue

                            solver.setup(problem1)
                            res1_using_2 = solver.solve(res2_using_1.traj)
                            if res1_using_2.traj is None:
                                continue

                            average: float = 0.5 * (res2_using_1.n_call + res1_using_2.n_call)
                            print(
                                "1->2: {}, 2->1: {}".format(
                                    res2_using_1.n_call, res1_using_2.n_call
                                )
                            )
                            file_path = self.dump_path / "{}.pkl".format(uuid.uuid4())
                            with file_path.open(mode="wb") as f:
                                pickle.dump((task1, task2, average), f)
                            n_saved += 1
                            pbar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-domain", type=str, default="tbrr_rrt", help="")
    args = parser.parse_args()
    domain_name: str = args.domain

    lib = load_library(domain_name, "cpu", limit_thread=True)
    domain: DomainProvider = DomainSelector[domain_name].value

    n_sample = 1000
    n_process = 12

    n_alloc_list = split_number(n_sample, n_process)

    pp = Path("./tmp_result")
    pp.mkdir(exist_ok=True)

    process_list = []
    for i in range(n_process):
        sampler = Sampler(lib, domain, n_alloc_list[i], pp, i == 0)
        p = Process(target=sampler.run, args=())
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    raw_data = []
    for file_path in pp.iterdir():
        with file_path.open(mode="rb") as f:
            task1, task2, cost = pickle.load(f)
            x1 = task1.descriptions[0][1]
            x2 = task2.descriptions[0][1]
            raw_data.append((x1, x2, cost))

    ppp = Path("dataset.pkl")
    with ppp.open(mode="wb") as f:
        pickle.dump(raw_data, f)
