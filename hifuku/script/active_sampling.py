import warnings

import skplan
from mohou.file import get_project_path
from mohou.trainer import TrainConfig
from mohou.utils import log_package_version_info

import hifuku
from hifuku.library import ClusterBasedSolutionLibrarySampler, LibrarySamplerConfig
from hifuku.neuralnet import VoxelAutoEncoder, VoxelAutoEncoderConfig
from hifuku.threedim.tabletop import TabletopPlanningProblem
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

if __name__ == "__main__":
    pp = get_project_path("tabletop_solution_library")
    pp.mkdir(exist_ok=True)
    logger = create_default_logger(pp, "library_gen")
    log_package_version_info(logger, hifuku)
    log_package_version_info(logger, skplan)

    ae_model = VoxelAutoEncoder(VoxelAutoEncoderConfig())
    # lconfig = LibrarySamplerConfig(
    #    n_problem=1000,
    #    n_problem_inner=50,
    #    train_config=TrainConfig(n_epoch=40),
    #    n_solution_candidate=10,
    #    n_difficult_problem=100,
    #    solvable_threshold_factor=0.6,
    # )  # all pass
    lconfig = LibrarySamplerConfig(
        n_problem=1000,
        n_problem_inner=50,
        train_config=TrainConfig(n_epoch=40),
        n_solution_candidate=10,
        n_difficult_problem=100,
        solvable_threshold_factor=0.6,
    )  # all pass
    lib_sampler = ClusterBasedSolutionLibrarySampler.initialize(
        TabletopPlanningProblem, ae_model, lconfig, use_distributed=True
    )

    for i in range(50):
        print(i)
        lib_sampler.step_active_sampling(pp)
