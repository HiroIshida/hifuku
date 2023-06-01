import argparse
from pathlib import Path
from typing import Optional

import rpbench
import selcol
import skmp
from mohou.file import get_project_path
from mohou.trainer import TrainCache, TrainConfig
from mohou.utils import log_package_version_info

import hifuku
from hifuku.domain import select_domain
from hifuku.library import LibrarySamplerConfig, SimpleSolutionLibrarySampler
from hifuku.neuralnet import AutoEncoderBase
from hifuku.script_utils import load_compatible_autoencoder, load_library
from hifuku.utils import create_default_logger, filter_warnings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", type=str, default="tbrr_sqp", help="")
    parser.add_argument("-aepn", type=str, help="auto encoder project name")
    parser.add_argument("-aepp", type=str, help="auto encoder project path")
    parser.add_argument("--warm", action="store_true", help="warm start")
    parser.add_argument("--lm", action="store_true", help="use light weight nn")

    args = parser.parse_args()
    domain_name: str = args.type
    warm_start: bool = args.warm
    auto_encoder_project_name: Optional[str] = args.aepn
    auto_encoder_project_path: Optional[str] = args.aepp
    use_light_model: bool = args.lm

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
    )  # all pass

    # load auto encoder
    ae_project_path: Optional[Path]
    if auto_encoder_project_name is not None:
        ae_project_path = get_project_path(auto_encoder_project_name)
    elif auto_encoder_project_path is not None:
        ae_project_path = Path(auto_encoder_project_path).expanduser()
    else:
        ae_project_path = None

    ae_model: AutoEncoderBase
    if ae_project_path is None:
        ae_model = load_compatible_autoencoder(domain_name)
    else:
        ae_model = TrainCache.load_all(ae_project_path)[0].best_model

    lib_sampler = SimpleSolutionLibrarySampler.initialize(
        domain.task_type,
        domain.solver_type,
        domain.solver_config,
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
