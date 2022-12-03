import argparse
import warnings
from enum import Enum

import skplan
from mohou.file import get_project_path
from mohou.trainer import TrainCache, TrainConfig
from mohou.utils import log_package_version_info

import hifuku
from hifuku.library import LibrarySamplerConfig, SimpleSolutionLibrarySampler
from hifuku.neuralnet import VoxelAutoEncoder
from hifuku.threedim.tabletop import (
    CachedProblemPool,
    TabletopMeshProblem,
    TabletopPlanningProblem,
    VoxbloxTabletopMeshProblem,
    VoxbloxTabletopPlanningProblem,
)
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


class ProblemType(Enum):
    normal = (TabletopMeshProblem, TabletopPlanningProblem)
    voxblox = (VoxbloxTabletopMeshProblem, VoxbloxTabletopPlanningProblem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", type=str, default="normal", help="")
    parser.add_argument("--cache", action="store_true", help="use cache")
    args = parser.parse_args()
    use_cache: bool = args.cache
    mesh_type_name: str = args.type

    mesh_problem_t, planning_problem_t = ProblemType[mesh_type_name].value
    pp_mesh = get_project_path("tabletop_mesh-{}".format(mesh_problem_t.__name__))
    pp = get_project_path("tabletop_solution_library-{}".format(planning_problem_t.__name__))
    pp.mkdir(exist_ok=True)

    logger = create_default_logger(pp, "library_gen")
    log_package_version_info(logger, hifuku)
    log_package_version_info(logger, skplan)

    if use_cache:
        pool_single = CachedProblemPool.load(
            planning_problem_t, mesh_problem_t, 1, pp_mesh / "cache"
        )
    else:
        pool_single = None

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
        planning_problem_t, ae_model, lconfig, pool_single=pool_single, use_distributed=True
    )

    for i in range(50):
        print(i)
        lib_sampler.step_active_sampling(pp)
