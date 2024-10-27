import argparse
import multiprocessing
import resource
from pathlib import Path
from typing import Dict, Literal, Optional

import psutil
import torch
import yaml
from mohou.trainer import TrainConfig

from hifuku.core import LibrarySamplerConfig, SimpleSolutionLibrarySampler
from hifuku.domain import select_domain
from hifuku.script_utils import (
    create_default_logger,
    filter_warnings,
    get_project_path,
    load_compatible_autoencoder,
    watch_memory,
)


def parse_config_yaml(dic: Dict) -> LibrarySamplerConfig:
    if "train_config" in dic:
        train_config = TrainConfig(**dic["train_config"])
        dic["train_config"] = train_config
    ls_config = LibrarySamplerConfig(**dic)
    return ls_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", type=str, default="tbrr_sqp", help="")
    parser.add_argument("-n", type=int, default=100, help="")
    parser.add_argument("-n_limit_batch", type=int, help="", default=50000)
    parser.add_argument("-conf", type=str)
    parser.add_argument("-post", type=str)
    parser.add_argument("--warm", action="store_true", help="warm start")
    parser.add_argument("--untrained", action="store_true", help="use untrained autoencoder")
    parser.add_argument("--local", action="store_true", help="don't use distributed computers")
    parser.add_argument("--hour", type=int, default=24, help="time limit in hours")

    args = parser.parse_args()
    domain_name: str = args.type
    warm_start: bool = args.warm
    n_step: int = args.n
    use_distributed: bool = not args.local
    use_pretrained_ae: bool = not args.untrained
    library_sampling_conf_path_str: Optional[str] = args.conf
    project_name_postfix: Optional[str] = args.post

    # first of all, we check that cpu-affinity of root process is set properly
    aff_root_process = psutil.Process(1).cpu_affinity()
    n_logical = psutil.cpu_count(logical=True)
    aff_isolated = set(list(range(n_logical))) - set(aff_root_process)
    assert (
        len(aff_isolated) <= 2
    )  # we may use 2 logical core for benchmarking but we must keep other cores free

    # require explicitly setting to None
    assert project_name_postfix is not None
    if project_name_postfix == "none":
        project_name_postfix = None

    # if postfix can be interpreted as float, then store it to acceptable_fp variable
    try:
        acceptable_fp = float(project_name_postfix)
    except ValueError:
        acceptable_fp = None

    torch.backends.cudnn.enabled = False
    assert torch.cuda.is_available()

    filter_warnings()
    domain = select_domain(domain_name)
    project_path = get_project_path(domain_name, project_name_postfix)

    logger = create_default_logger(project_path, "library_gen")

    logger.info("cmd args: {}".format(args))

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
        if not use_pretrained_ae:
            assert lsconfig.train_with_encoder
    assert lsconfig.acceptable_false_positive_rate == acceptable_fp
    logger.info("lsconfig: {}".format(lsconfig))

    if not use_pretrained_ae:
        assert (
            lsconfig.train_with_encoder
        ), "you must train encoder, otherwise encoder will be just a random one"

    # run memory watchdog
    p_watchdog = multiprocessing.Process(target=watch_memory, args=(5.0,))
    p_watchdog.start()

    n_grid = None
    if not use_pretrained_ae:
        task_type = domain.task_type
        task = task_type.sample()
        mat = task.export_task_expression(use_matrix=True).get_matrix()
        n_grid: Optional[Literal[56, 112]] = mat.shape[0]
    ae_model = load_compatible_autoencoder(domain_name, use_pretrained_ae, n_grid)

    lib_sampler = SimpleSolutionLibrarySampler.initialize(
        domain.task_type,
        domain.solver_type,
        domain.solver_config,
        ae_model,
        lsconfig,
        project_path,
        use_distributed=use_distributed,
        n_limit_batch_solver=args.n_limit_batch,
        warm_start=warm_start,
    )

    for i in range(n_step):
        profs = lib_sampler.sampler_history.elapsed_time_history
        if len(profs) > 0:
            time_total = sum([info.t_total for info in profs])  # type: ignore[misc]
            logger.info("time_total: {}".format(time_total))
            if time_total > 3600 * args.hour:
                logger.info(f"time_total > {args.hour} hours, break")
                break

        sampling_successful = lib_sampler.step_active_sampling()

        latest_est = lib_sampler.sampler_history.coverage_est_history[-1]
        if latest_est > 0.98:
            logger.info("coverage > 0.98, break")
            break

    p_watchdog.terminate()
    p_watchdog.join()
