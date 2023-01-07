import argparse
import warnings
from enum import Enum

import skplan
from mohou.file import get_project_path
from mohou.trainer import TrainCache, TrainConfig
from mohou.utils import log_package_version_info

import hifuku
from hifuku.domain import DomainProvider, TBRR_SQP_DomainProvider
from hifuku.library import LibrarySamplerConfig, SimpleSolutionLibrarySampler
from hifuku.neuralnet import VoxelAutoEncoder
from hifuku.utils import create_default_logger

warnings.filterwarnings("ignore", message="Values in x were outside bounds during")
warnings.filterwarnings("ignore", message="texture specified in URDF is not supported")
warnings.filterwarnings("ignore", message="Converting sparse A to a CSC")
warnings.filterwarnings("ignore", message="urllib3")
warnings.filterwarnings(
    "ignore",
    message="undefined symbol: _ZNK3c1010TensorImpl36is_contiguous_nondefault_policy_implENS_12MemoryFormatE",
)
warnings.filterwarnings(
    "ignore", message="`np.float` is a deprecated alias for the builtin `float`"
)


class DomainType(Enum):
    normal = TBRR_SQP_DomainProvider


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", type=str, default="normal", help="")
    args = parser.parse_args()
    mesh_type_name: str = args.type

    domain: DomainProvider = DomainType[mesh_type_name].value
    mesh_sampler_type = TBRR_SQP_DomainProvider.get_compat_mesh_sampler_type()
    domain_name = TBRR_SQP_DomainProvider.get_domain_name()

    pp_mesh = get_project_path("hifuku-{}".format(mesh_sampler_type.__name__))
    pp = get_project_path("tabletop_solution_library-{}".format(domain_name))
    pp.mkdir(exist_ok=True)

    logger = create_default_logger(pp, "library_gen")
    log_package_version_info(logger, hifuku)
    log_package_version_info(logger, skplan)

    ae_model = TrainCache.load_latest(pp_mesh, VoxelAutoEncoder).best_model
    lconfig = LibrarySamplerConfig(
        n_problem=3000,
        n_problem_inner=200,
        train_config=TrainConfig(n_epoch=40),
        n_solution_candidate=10,
        n_difficult_problem=100,
        solvable_threshold_factor=0.6,
        acceptable_false_positive_rate=0.01,
    )  # all pass
    lib_sampler = SimpleSolutionLibrarySampler.initialize(
        TBRR_SQP_DomainProvider.get_task_type(),
        TBRR_SQP_DomainProvider.get_solver_type(),
        TBRR_SQP_DomainProvider.get_solver_config(),
        ae_model,
        lconfig,
        pool_single=None,
        use_distributed=True,
    )

    for i in range(50):
        print(i)
        lib_sampler.step_active_sampling(pp)
