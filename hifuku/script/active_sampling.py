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
    TBRR_RRT_DomainProvider,
    TBRR_SQP_DomainProvider,
)
from hifuku.library import (
    LibrarySamplerConfig,
    SimpleSolutionLibrarySampler,
    SolutionLibrary,
)
from hifuku.neuralnet import AutoEncoderBase, NullAutoEncoder, VoxelAutoEncoder
from hifuku.utils import create_default_logger, filter_warnings


class DomainType(Enum):
    tbrr_sqp = TBRR_SQP_DomainProvider
    tbrr_rrt = TBRR_RRT_DomainProvider
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

    if mesh_sampler_type is None:
        ae_model: AutoEncoderBase = NullAutoEncoder()
    else:
        ae_pp = get_project_path("hifuku-{}".format(mesh_sampler_type.__name__))
        ae_model = TrainCache.load(ae_pp, VoxelAutoEncoder).best_model

    pp = get_project_path("tabletop_solution_library-{}".format(domain_name))
    pp.mkdir(exist_ok=True)

    logger = create_default_logger(pp, "library_gen")
    log_package_version_info(logger, hifuku)
    log_package_version_info(logger, rpbench)
    log_package_version_info(logger, skmp)
    log_package_version_info(logger, selcol)

    lconfig = LibrarySamplerConfig(
        n_problem=10000,
        n_problem_inner=50,
        train_config=TrainConfig(n_epoch=100),
        n_solution_candidate=30,
        n_difficult_problem=300,
        solvable_threshold_factor=1.0,
        difficult_threshold_factor=1.0,
        acceptable_false_positive_rate=0.03,
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
    # solver=MultiProcessBatchProblemSolver(
    #     domain.get_solver_type(), domain.get_solver_config(), 2
    # ),
    if warm_start:
        lib = SolutionLibrary.load(
            pp, domain.get_task_type(), domain.get_solver_type(), torch.device("cuda")
        )[0]
        lib.limit_thread = True
        lib_sampler.library = lib

    for i in range(100):
        print(i)
        lib_sampler.step_active_sampling(pp)
