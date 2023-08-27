import logging
import time
from pathlib import Path
from typing import Literal, Optional, Type, Union

import mohou.file
import psutil
import torch
from mohou.trainer import TrainCache

from hifuku.domain import DomainProtocol, select_domain
from hifuku.library import SolutionLibrary
from hifuku.neuralnet import AutoEncoderBase, AutoEncoderConfig, NullAutoEncoder

logger = logging.getLogger(__name__)


def load_compatible_autoencoder(
    domain: Union[str, Type[DomainProtocol]], pretrained: bool
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
        return ae_model
    else:
        logger.info("use untrained autoencoder")
        T = domain.auto_encoder_type
        if issubclass(T, NullAutoEncoder):
            logger.info("actually, we will use null autoencoder")
            return NullAutoEncoder()
        else:
            conf = AutoEncoderConfig()
            logger.info("Initialize {} with default config {}".format(T.__name__, conf))
            return T(conf)  # type: ignore


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
    limit_thread: bool = False,
    project_path: Optional[Path] = None,
    postfix: Optional[str] = None,
) -> SolutionLibrary:

    if isinstance(domain, str):
        domain = select_domain(domain)

    if project_path is None:
        project_path = get_project_path(domain, postfix)
    lib = SolutionLibrary.load(
        project_path, domain.task_type, domain.solver_type, torch.device(device)
    )[0]
    lib.limit_thread = limit_thread
    return lib


def watch_memmory(interval: float, debug: bool = True):
    while True:
        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        if debug:
            logger.debug("memmory usage: {}%".format(ram_percent))
        else:
            logger.info("memmory usage: {}%".format(ram_percent))
        time.sleep(interval)
