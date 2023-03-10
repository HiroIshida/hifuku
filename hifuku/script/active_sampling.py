import argparse

import rpbench
import selcol
import skmp
from mohou.trainer import TrainConfig
from mohou.utils import log_package_version_info

import hifuku
from hifuku.library import LibrarySamplerConfig, SimpleSolutionLibrarySampler
from hifuku.script_utils import (
    DomainSelector,
    get_project_path,
    load_compatible_autoencoder,
    load_library,
)
from hifuku.utils import create_default_logger, filter_warnings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", type=str, default="tbrr_sqp", help="")
    parser.add_argument("--warm", action="store_true", help="warm start")

    args = parser.parse_args()
    domain_name: str = args.type
    warm_start: bool = args.warm

    filter_warnings()
    domain = DomainSelector[domain_name].value
    pp = get_project_path(domain_name)

    logger = create_default_logger(pp, "library_gen")
    log_package_version_info(logger, hifuku)
    log_package_version_info(logger, rpbench)
    log_package_version_info(logger, skmp)
    log_package_version_info(logger, selcol)

    lconfig = LibrarySamplerConfig(
        n_problem=10000,
        n_problem_inner=200,
        train_config=TrainConfig(n_epoch=100),
        n_solution_candidate=100,
        n_difficult_problem=500,
        solvable_threshold_factor=1.0,
        difficult_threshold_factor=1.0,
        acceptable_false_positive_rate=0.03,
    )  # all pass

    ae_model = load_compatible_autoencoder(domain_name)
    lib_sampler = SimpleSolutionLibrarySampler.initialize(
        domain.get_task_type(),
        domain.get_solver_type(),
        domain.get_solver_config(),
        ae_model,
        lconfig,
        pool_single=None,
        use_distributed=True,
        reuse_cached_validation_set=warm_start,
    )

    if warm_start:
        lib_sampler.library = load_library(domain_name, "cuda", True)

    for i in range(100):
        print(i)
        lib_sampler.step_active_sampling(pp)
