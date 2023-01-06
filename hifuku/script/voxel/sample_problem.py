import argparse
import multiprocessing
import os
import uuid
import warnings
from enum import Enum

import numpy as np
import tqdm
from mohou.file import get_project_path
from mohou.utils import log_package_version_info

import hifuku
from hifuku.datagen import MultiProcessBatchProblemSampler
from hifuku.pool import TrivialProblemPool
from hifuku.task_wrap import TabletopBoxWorldWrap, TabletopVoxbloxBoxWorldWrap
from hifuku.utils import create_default_logger

warnings.filterwarnings("ignore", message="Values in x were outside bounds during")
warnings.filterwarnings("ignore", message="texture specified in URDF is not supported")

np.random.seed(0)


class ProblemType(Enum):
    normal = TabletopBoxWorldWrap
    voxblox = TabletopVoxbloxBoxWorldWrap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", type=str, default="normal", help="")
    args = parser.parse_args()
    mesh_type_name: str = args.type

    problem_type = ProblemType[mesh_type_name].value  # type: ignore

    pp = get_project_path("tabletop_mesh-{}".format(problem_type.__name__))
    logger = create_default_logger(pp, "mesh_generation")
    log_package_version_info(logger, hifuku)
    cache_base_path = pp / "cache"
    cache_base_path.mkdir(exist_ok=True, parents=True)
    n_cpu: int = os.cpu_count()  # type: ignore
    n_process = int(0.5 * n_cpu)
    n_problem = 100
    for _ in range(1):
        # sampler = DistributeBatchProblemSampler()  # type: ignore
        sampler = MultiProcessBatchProblemSampler()  # type: ignore
        pool = TrivialProblemPool(problem_type, n_problem_inner=0)
        problems = sampler.sample_batch(n_problem, pool.as_predicated())

        def f(args):
            problems, base_path, disable_tqdm = args
            for problem in tqdm.tqdm(problems, disable=disable_tqdm):
                p = base_path / (str(uuid.uuid4()) + ".pkl")
                problem.dump(p)

        args_list = []
        indices_list = np.array_split(np.arange(n_problem), n_process)  # type: ignore
        for idx, indices in enumerate(indices_list):
            disable_tqdm = idx != 0
            problems_part = [problems[i] for i in indices]
            args = ([problems[i] for i in indices], cache_base_path, disable_tqdm)  # type: ignore
            args_list.append(args)

        with multiprocessing.Pool(n_process) as p:
            p.map(f, args_list)
