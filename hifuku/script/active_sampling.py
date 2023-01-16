import argparse
from enum import Enum

import rpbench
import selcol
import skmp
import torch
from mohou.file import get_project_path
from mohou.trainer import TrainCache, TrainConfig
from mohou.utils import log_package_version_info

import hifuku
from hifuku.domain import (
    DomainProvider,
    TBDR_SQP_DomainProvider,
    TBRR_SQP_DomainProvider,
)
from hifuku.library import (
    LibrarySamplerConfig,
    SimpleSolutionLibrarySampler,
    SolutionLibrary,
)
from hifuku.neuralnet import VoxelAutoEncoder
from hifuku.utils import create_default_logger, filter_warnings


class DomainType(Enum):
    tbrr_sqp = TBRR_SQP_DomainProvider
    tbdr_sqp = TBDR_SQP_DomainProvider


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", type=str, default="tbrr_sqp", help="")
    parser.add_argument("--warm", action="store_true", help="warm start")

    args = parser.parse_args()
    mesh_type_name: str = args.type
    warm_start: bool = args.warm

    filter_warnings()

    domain: DomainProvider = DomainType[mesh_type_name].value
    mesh_sampler_type = domain.get_compat_mesh_sampler_type()
    domain_name = domain.get_domain_name()

    pp_mesh = get_project_path("hifuku-{}".format(mesh_sampler_type.__name__))
    pp = get_project_path("tabletop_solution_library-{}".format(domain_name))
    pp.mkdir(exist_ok=True)

    logger = create_default_logger(pp, "library_gen")
    log_package_version_info(logger, hifuku)
    log_package_version_info(logger, rpbench)
    log_package_version_info(logger, skmp)
    log_package_version_info(logger, selcol)

    ae_model = TrainCache.load_latest(pp_mesh, VoxelAutoEncoder).best_model
    lconfig = LibrarySamplerConfig(
        n_problem=1000,
        n_problem_inner=100,
        train_config=TrainConfig(n_epoch=40),
        n_solution_candidate=10,
        n_difficult_problem=100,
        solvable_threshold_factor=0.6,
        acceptable_false_positive_rate=0.01,
    )  # all pass
    lib_sampler = SimpleSolutionLibrarySampler.initialize(
        domain.get_task_type(),
        domain.get_solver_type(),
        domain.get_solver_config(),
        ae_model,
        lconfig,
        pool_single=None,
        use_distributed=True,
    )

    if warm_start:
        lib = SolutionLibrary.load(
            pp, domain.get_task_type(), domain.get_solver_type(), torch.device("cuda")
        )[0]
        lib.limit_thread = True
        lib_sampler.library = lib

    for i in range(50):
        print(i)
        lib_sampler.step_active_sampling(pp)
