import logging
import time
import warnings
from logging import Logger
from pathlib import Path
from typing import Literal, Optional, Type, Union

import mohou.file
import psutil
import torch
from mohou.trainer import TrainCache
from mohou.utils import detect_device, log_package_version_info

from hifuku.core import ActiveSamplerHistory, SolutionLibrary
from hifuku.domain import DomainProtocol, select_domain
from hifuku.neuralnet import AutoEncoderBase, AutoEncoderConfig, NullAutoEncoder

logger = logging.getLogger(__name__)


def load_compatible_autoencoder(
    domain: Union[str, Type[DomainProtocol]],
    pretrained: bool,
    n_grid: Optional[Literal[56, 112]] = None,
) -> AutoEncoderBase:
    if isinstance(domain, str):
        domain = select_domain(domain)

    if pretrained:
        logger.info("use pretrained autoencoder")
        if domain.auto_encoder_project_name is None:
            logger.info("actually, we will use null autoencoder")
            ae_model: AutoEncoderBase = NullAutoEncoder()
        else:
            ae_pp = mohou.file.get_project_path(domain.auto_encoder_project_name)
            ae_model = TrainCache.load_all(ae_pp)[0].best_model
            logger.info("load from {}".format(ae_pp))
            assert isinstance(ae_model, AutoEncoderBase)
        ae_model.put_on_device(detect_device())
        return ae_model
    else:
        logger.info("use untrained autoencoder")
        T = domain.auto_encoder_type
        if issubclass(T, NullAutoEncoder):
            logger.info("actually, we will use null autoencoder")
            ae = NullAutoEncoder()
        else:
            assert n_grid is not None
            conf = AutoEncoderConfig(n_grid=n_grid)
            logger.info("Initialize {} with default config {}".format(T.__name__, conf))
            ae = T(conf)  # type: ignore
        ae.put_on_device(detect_device())
        return ae_model


def get_project_path(
    domain: Union[str, Type[DomainProtocol]], postfix: Optional[str] = None
) -> Path:

    if isinstance(domain, str):
        domain = select_domain(domain)

    domain_identifier = domain.get_domain_name()
    if postfix is None:
        project_name = "hifuku-{}".format(domain_identifier)
    else:
        project_name = "hifuku-{}-{}".format(domain_identifier, postfix)
    pp = mohou.file.get_project_path(project_name)
    pp.mkdir(exist_ok=True)
    return pp


def load_library(
    domain: Union[str, Type[DomainProtocol]],
    device: Literal["cpu", "cuda"],
    project_path: Optional[Path] = None,
    postfix: Optional[str] = None,
) -> SolutionLibrary:

    if isinstance(domain, str):
        domain = select_domain(domain)

    if project_path is None:
        project_path = get_project_path(domain, postfix)
    lib = SolutionLibrary.load(project_path, torch.device(device))
    return lib


def load_sampler_history(
    domain: Union[str, Type[DomainProtocol]],
    project_path: Optional[Path] = None,
    postfix: Optional[str] = None,
) -> ActiveSamplerHistory:

    if isinstance(domain, str):
        domain = select_domain(domain)
    if project_path is None:
        project_path = get_project_path(domain, postfix)
    state = ActiveSamplerHistory.load(project_path)
    return state


def watch_memory(interval: float, debug: bool = True):
    while True:
        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        if debug:
            logger.debug("memory usage: {}%".format(ram_percent))
        else:
            logger.info("memory usage: {}%".format(ram_percent))
        time.sleep(interval)


def create_default_logger(
    project_path: Optional[Path], prefix: str, stream_level: int = logging.INFO
) -> Logger:
    logger = logging.getLogger()  # root logger
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(stream_level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if project_path is not None:
        # create file handler
        timestr = "_" + time.strftime("%Y%m%d%H%M%S")
        log_dir_path = project_path / "log"
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir_path / (prefix + timestr + ".log")

        fh = logging.FileHandler(str(log_file_path))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)

        logger.addHandler(fh)

        log_sym_path = log_dir_path / ("latest_" + prefix + ".log")

        logger.info("create log symlink :{0} => {1}".format(log_file_path, log_sym_path))
        if log_sym_path.is_symlink():
            log_sym_path.unlink()
        log_sym_path.symlink_to(log_file_path)

    import rpbench
    import skmp

    import hifuku

    log_package_version_info(logger, hifuku)
    log_package_version_info(logger, rpbench)
    log_package_version_info(logger, skmp)

    return logger


def filter_warnings():
    warnings.filterwarnings("ignore", message="Values in x were outside bounds during")
    warnings.filterwarnings("ignore", message="texture specified in URDF is not supported")
    warnings.filterwarnings("ignore", message="Converting sparse A to a CSC")
    warnings.filterwarnings("ignore", message="urllib3")
    warnings.filterwarnings(
        "ignore",
        message="undefined symbol: _ZNK3c1010TensorImpl36is_contiguous_nondefault_policy_implENS_12MemoryFormatE",
    )
