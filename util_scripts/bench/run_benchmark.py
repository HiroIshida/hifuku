import argparse
import pickle
import time
from pathlib import Path

import torch
from compatible_solver import CompatibleSolvers
from mohou.file import get_project_path
from rpbench.interface import PlanningDataset

from hifuku.domain import select_domain
from hifuku.library import LibraryBasedGuaranteedSolver, SolutionLibrary
from hifuku.utils import create_default_logger, filter_warnings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-domain", type=str, default="humanoid_trr_sqp", help="")
    parser.add_argument("-n", type=int, default=300, help="")
    parser.add_argument("--feasible", action="store_true", help="use only feasible problem set")
    parser.add_argument("--dataset", action="store_true", help="load dataset")
    parser.add_argument("--proposed", action="store_true", help="use only proposed")
    args = parser.parse_args()

    n_sample = args.n
    only_feasible: bool = args.feasible
    load_dataset: bool = args.dataset
    domain_name: str = args.domain
    domain = select_domain(domain_name)
    pp = get_project_path(domain_name)
    task_type = domain.task_type

    filter_warnings()

    logger = create_default_logger(Path("./").expanduser(), "run_benchmark")
    logger.info(args)

    if only_feasible:
        logger.info("use feasible problem set")
        problem_set_path = Path("./problem_set") / (task_type.__name__ + ".pkl")
        with problem_set_path.open(mode="rb") as f:
            pairs = pickle.load(f)
            tasks_full, resultss = zip(*pairs)
            for task, traj in zip(tasks_full, resultss):
                assert traj is not None
        assert n_sample <= len(tasks_full), "available: {}".format(len(tasks_full))
        tasks = tasks_full[:n_sample]
    else:
        logger.info("sample problem set now")
        tasks = [task_type.sample(1) for _ in range(n_sample)]
    assert tasks[0].n_inner_task == 1

    if load_dataset:
        pairs_path = Path("./raw_library") / (task_type.__name__ + ".pkl")
        with pairs_path.open(mode="rb") as f:
            pairs_tmp = pickle.load(f)
        pairs = []
        for task, traj in pairs_tmp:
            pairs.append((task, traj))
        dataset = PlanningDataset(pairs, task_type, time.time())
    else:
        dataset = None
    if args.proposed:
        solver_table = {}
    else:
        solver_table = CompatibleSolvers.get_compatible_solvers(task_type.__name__, dataset=dataset)

    # setup proposed solver
    pp = get_project_path("hifuku-{}".format(domain.get_domain_name()))
    libraries = SolutionLibrary.load(pp, domain.task_type, domain.solver_type, torch.device("cpu"))
    lib = libraries[0]
    proposed = LibraryBasedGuaranteedSolver.init(lib)
    proposed_name = "proposed-guaranteed"
    solver_table["proposed-guaranteed"] = proposed

    results = []
    fp_count = 0
    fp_denominator = 0
    count_dict = {key: 0 for key in solver_table.keys()}
    tasks_dump = []

    for i, task in enumerate(tasks):
        logger.info("task num: {}".format(i))

        result = {}
        for name, solver in solver_table.items():
            logger.info("solver name: {}".format(name))
            solver.setup(task)
            res = solver.solve()
            logger.info("solved?: {}, time: {}".format(res.traj is not None, res.time_elapsed))

            if res.traj is not None:
                count_dict[name] += 1

            result[name] = res

            if name == proposed_name:
                if res.time_elapsed is not None:
                    fp_denominator += 1
                    if res.traj is None:
                        fp_count += 1

        logger.info("count dict: {}".format(count_dict))
        tasks_dump.append(task)
        results.append(result)

        if fp_denominator > 0:
            fp_rate = fp_count / fp_denominator
            logger.info("current fp rate: {}".format(fp_rate))

    result_base_path = Path("./result")
    result_path: Path = result_base_path / "result-{}".format(domain.get_domain_name())
    with result_path.open(mode="wb") as f:
        pickle.dump((tasks_dump, results), f)
