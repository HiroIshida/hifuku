from pathlib import Path
from typing import Literal, Optional, Type, Union

import mohou.file
import torch
from mohou.trainer import TrainCache

from hifuku.domain import DomainProtocol, select_domain
from hifuku.library import SolutionLibrary
from hifuku.neuralnet import AutoEncoderBase, NullAutoEncoder, VoxelAutoEncoder


def load_compatible_autoencoder(domain: Union[str, Type[DomainProtocol]]) -> AutoEncoderBase:
    if isinstance(domain, str):
        domain = select_domain(domain)
    if domain.mesh_sampler_type is None:
        ae_model: AutoEncoderBase = NullAutoEncoder()
    else:
        ae_pp = mohou.file.get_project_path("hifuku-{}".format(domain.mesh_sampler_type.__name__))
        ae_model = TrainCache.load(ae_pp, VoxelAutoEncoder).best_model
    return ae_model


def get_project_path(domain: Union[str, Type[DomainProtocol]]) -> Path:

    if isinstance(domain, str):
        domain = select_domain(domain)

    domain_identifier = domain.get_domain_name()
    pp = mohou.file.get_project_path("tabletop_solution_library-{}".format(domain_identifier))
    pp.mkdir(exist_ok=True)
    return pp


def load_library(
    domain: Union[str, Type[DomainProtocol]],
    device: Literal["cpu", "cuda"],
    limit_thread: bool = False,
    project_path: Optional[Path] = None,
) -> SolutionLibrary:

    if isinstance(domain, str):
        domain = select_domain(domain)

    if project_path is None:
        project_path = get_project_path(domain)
    lib = SolutionLibrary.load(
        project_path, domain.task_type, domain.solver_type, torch.device(device)
    )[0]
    lib.limit_thread = limit_thread
    return lib
