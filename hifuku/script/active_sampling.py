import argparse
import multiprocessing
import resource
from pathlib import Path
from typing import Dict, Optional

import rpbench
import skmp
import yaml
from mohou.trainer import TrainConfig
from mohou.utils import log_package_version_info

import hifuku
from hifuku.domain import select_domain
from hifuku.library import LibrarySamplerConfig, SimpleSolutionLibrarySampler
from hifuku.script_utils import (
    get_project_path,
    load_compatible_autoencoder,
    load_library,
    watch_memmory,
)
from hifuku.utils import create_default_logger, filter_warnings


def parse_config_yaml(dic: Dict) -> LibrarySamplerConfig:
    train_config = TrainConfig(**dic["train_config"])
    dic["train_config"] = train_config
    ls_config = LibrarySamplerConfig(**dic)
    return ls_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", type=str, default="tbrr_sqp", help="")
    parser.add_argument("-n", type=int, default=100, help="")
    parser.add_argument("-conf", type=str)
    parser.add_argument("--warm", action="store_true", help="warm start")
    parser.add_argument("--fptest", action="store_true", help="test false positive rate")
    parser.add_argument("--local", action="store_true", help="don't use distributed computers")

    args = parser.parse_args()
    domain_name: str = args.type
    warm_start: bool = args.warm
    test_fp_rate: bool = args.fptest
    n_step: int = args.n
    use_distributed: bool = not args.local
    library_sampling_conf_path_str: Optional[str] = args.conf

    filter_warnings()
    domain = select_domain(domain_name)
    project_path = get_project_path(domain_name)

    logger = create_default_logger(project_path, "library_gen")
    log_package_version_info(logger, hifuku)
    log_package_version_info(logger, rpbench)
    log_package_version_info(logger, skmp)
    # log_package_version_info(logger, selcol)

    # set file open limit to large
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    logger.info("current NOFILE value: (soft: {}, hard: {})".format(soft, hard))
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    logger.info("set NOFILE to {}".format(hard))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    logger.info("current NOFILE value: (soft: {}, hard: {})".format(soft, hard))

    if library_sampling_conf_path_str is None:
        library_sampling_conf_path = project_path / "lsconf.yaml"
    else:
        library_sampling_conf_path = Path(library_sampling_conf_path_str).expanduser()
    with library_sampling_conf_path.open(mode="r") as f:
        dic = yaml.safe_load(f)
        lsconfig = parse_config_yaml(dic)
    logger.info("lsconfig: {}".format(lsconfig))

    # run memmory watchdog
    p_watchdog = multiprocessing.Process(target=watch_memmory, args=(5.0,))
    p_watchdog.start()

    ae_model = load_compatible_autoencoder(domain_name)
    lib_sampler = SimpleSolutionLibrarySampler.initialize(
        domain.task_type,
        domain.solver_type,
        domain.solver_config,
        ae_model,
        lsconfig,
        project_path,
        pool_single=None,
        use_distributed=use_distributed,
        reuse_cached_validation_set=warm_start,
        invalidate_gridsdf=True,
        test_false_positive_rate=test_fp_rate,
    )

    if warm_start:
        lib_sampler.library = load_library(domain_name, "cuda", True)

    for i in range(n_step):
        print(i)
        lib_sampler.step_active_sampling()

    p_watchdog.terminate()
    p_watchdog.join()
