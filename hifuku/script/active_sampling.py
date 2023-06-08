import argparse

import rpbench
import selcol
import skmp
from mohou.trainer import TrainConfig
from mohou.utils import log_package_version_info

import hifuku
from hifuku.domain import select_domain
from hifuku.library import LibrarySamplerConfig, SimpleSolutionLibrarySampler
from hifuku.script_utils import (
    get_project_path,
    load_compatible_autoencoder,
    load_library,
)
from hifuku.utils import create_default_logger, filter_warnings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", type=str, default="tbrr_sqp", help="")
    parser.add_argument("--warm", action="store_true", help="warm start")
    parser.add_argument("--lm", action="store_true", help="use light weight nn")
    parser.add_argument("--fptest", action="store_true", help="test false positive rate")

    args = parser.parse_args()
    domain_name: str = args.type
    warm_start: bool = args.warm
    use_light_model: bool = args.lm
    test_fp_rate: bool = args.fptest

    filter_warnings()
    domain = select_domain(domain_name)
    pp = get_project_path(domain_name)

    logger = create_default_logger(pp, "library_gen")
    log_package_version_info(logger, hifuku)
    log_package_version_info(logger, rpbench)
    log_package_version_info(logger, skmp)
    log_package_version_info(logger, selcol)

    if use_light_model:
        iterpred_model_config = {"n_layer1_width": 100, "n_layer2_width": 100, "n_layer3_width": 20}
    else:
        iterpred_model_config = None  # type: ignore

    lconfig = LibrarySamplerConfig(
        n_problem=10000,
        n_problem_inner=200,
        train_config=TrainConfig(n_epoch=100),
        n_solution_candidate=100,
        n_difficult_problem=500,
        solvable_threshold_factor=1.0,
        difficult_threshold_factor=1.0,
        acceptable_false_positive_rate=0.03,
        iterpred_model_config=iterpred_model_config,
        bootstrap_trial=1000,
    )  # all pass

    ae_model = load_compatible_autoencoder(domain_name)
    lib_sampler = SimpleSolutionLibrarySampler.initialize(
        domain.task_type,
        domain.solver_type,
        domain.solver_config,
        ae_model,
        lconfig,
        pool_single=None,
        use_distributed=True,
        reuse_cached_validation_set=warm_start,
        test_false_positive_rate=test_fp_rate,
    )

    if warm_start:
        lib_sampler.library = load_library(domain_name, "cuda", True)

    for i in range(100):
        print(i)
        lib_sampler.step_active_sampling(pp)
