from enum import Enum
from pathlib import Path
from typing import Literal

import mohou.file
import torch
from mohou.trainer import TrainCache

from hifuku.domain import (
    DomainProvider,
    TBDR_SQP_DomainProvider,
    TBRR_RRT_DomainProvider,
    TBRR_SQP_DomainProvider,
)
from hifuku.library import SolutionLibrary
from hifuku.neuralnet import AutoEncoderBase, NullAutoEncoder, VoxelAutoEncoder


class DomainSelector(Enum):
    tbrr_sqp = TBRR_SQP_DomainProvider
    tbrr_rrt = TBRR_RRT_DomainProvider
    tbdr_sqp = TBDR_SQP_DomainProvider


def load_compatible_autoencoder(domain_name: str) -> AutoEncoderBase:
    domain: DomainProvider = DomainSelector[domain_name].value
    mesh_sampler_type = domain.get_compat_mesh_sampler_type()
    if mesh_sampler_type is None:
        ae_model: AutoEncoderBase = NullAutoEncoder()
    else:
        ae_pp = mohou.file.get_project_path("hifuku-{}".format(mesh_sampler_type.__name__))
        ae_model = TrainCache.load(ae_pp, VoxelAutoEncoder).best_model
    return ae_model


def get_project_path(domain_name: str) -> Path:
    domain: DomainProvider = DomainSelector[domain_name].value
    domain_identifier = domain.get_domain_name()
    pp = mohou.file.get_project_path("tabletop_solution_library-{}".format(domain_identifier))
    pp.mkdir(exist_ok=True)
    return pp


def load_library(
    domain_name: str, device: Literal["cpu", "cuda"], limit_thread: bool = False
) -> SolutionLibrary:
    domain = DomainSelector[domain_name].value
    pp = get_project_path(domain_name)
    lib = SolutionLibrary.load(
        pp, domain.get_task_type(), domain.get_solver_type(), torch.device(device)
    )[0]
    lib.limit_thread = limit_thread
    return lib
