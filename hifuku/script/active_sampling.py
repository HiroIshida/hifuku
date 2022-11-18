import logging
import warnings

from mohou.file import get_project_path
from mohou.trainer import TrainConfig

from hifuku.guarantee.algorithm import (
    LibrarySamplerConfig,
    MultiProcessDatasetGenerator,
    SimpleFixedProblemPool,
    SolutionLibrarySampler,
)
from hifuku.neuralnet import VoxelAutoEncoder, VoxelAutoEncoderConfig
from hifuku.threedim.tabletop import TabletopPlanningProblem
from hifuku.utils import create_default_logger

warnings.filterwarnings("ignore", message="Values in x were outside bounds during")
warnings.filterwarnings("ignore", message="texture specified in URDF is not supported")
warnings.filterwarnings("ignore", message="Converting sparse A to a CSC")
warnings.filterwarnings(
    "ignore", message="`np.float` is a deprecated alias for the builtin `float`"
)

if __name__ == "__main__":
    pp = get_project_path("tabletop_solution_library")
    pp.mkdir(exist_ok=True)
    logger = create_default_logger(pp, "library_gen", logging.DEBUG)

    gen = MultiProcessDatasetGenerator(TabletopPlanningProblem)
    tconfig = TrainConfig(n_epoch=100)
    ae_model = VoxelAutoEncoder(VoxelAutoEncoderConfig())

    lconfig = LibrarySamplerConfig(
        n_problem=100,
        n_problem_inner=50,
        train_config=TrainConfig(n_epoch=100),
        n_solution_candidate=10,
        n_difficult_problem=100,
    )  # all pass
    validation_pool = SimpleFixedProblemPool.initialize(TabletopPlanningProblem, 10)
    lib_sampler = SolutionLibrarySampler.initialize(
        TabletopPlanningProblem, ae_model, gen, lconfig, validation_pool
    )

    for i in range(10):
        print(i)
        lib_sampler.step_active_sampling(pp)
