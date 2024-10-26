import time
from enum import Enum
from typing import ClassVar, Optional, Protocol, Type

import tqdm
from rpbench.articulated.jaxon.below_table import (
    HumanoidTableClutteredReachingTask,
    HumanoidTableClutteredReachingTask2,
    HumanoidTableReachingTask,
    HumanoidTableReachingTask2,
    HumanoidTableReachingTask3,
)
from rpbench.articulated.pr2.jskfridge import (
    JskFridgeReachingTask,
    JskFridgeVerticalReachingTask,
    JskFridgeVerticalReachingTask2,
)
from rpbench.articulated.pr2.minifridge import (
    FixedPR2MiniFridgeTask,
    PR2MiniFridgeTask,
    PR2MiniFridgeVoxelTask,
)
from rpbench.interface import TaskBase
from rpbench.two_dimensional.bubbly_world import (
    BubblyComplexMeshPointConnectTask,
    BubblyEmptyMeshPointConnectTask,
    BubblyModerateMeshPointConnectTask,
    BubblySimpleMeshPointConnectTask,
    DoubleIntegratorOptimizationSolver,
    DoubleIntegratorPlanningConfig,
    ParametricMazeTask1D,
    ParametricMazeTask2D,
    ParametricMazeTask3D,
    ParametricMazeTask4D,
    ParametricMazeTask5D,
    ParametricMazeSpecialTask
)
from rpbench.two_dimensional.dummy import (
    DummyConfig,
    DummyMeshTask,
    DummySolver,
    DummyTask,
    ProbDummyTask,
)
from skmp.solver.interface import AbstractScratchSolver, ConfigProtocol
from skmp.solver.nlp_solver.sqp_based_solver import SQPBasedSolver, SQPBasedSolverConfig
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig

from hifuku.datagen.batch_sampler import (
    DistributeBatchTaskSampler,
    MultiProcessBatchTaskSampler,
)
from hifuku.datagen.batch_solver import (
    DistributedBatchTaskSolver,
    MultiProcessBatchTaskSolver,
)
from hifuku.neuralnet import (
    AutoEncoderBase,
    NullAutoEncoder,
    PixelAutoEncoder,
    VoxelAutoEncoder,
)


class DomainProtocol(Protocol):
    task_type: ClassVar[Type[TaskBase]]
    solver_type: ClassVar[Type[AbstractScratchSolver]]
    solver_config: ClassVar[ConfigProtocol]
    auto_encoder_project_name: ClassVar[Optional[str]]
    auto_encoder_type: Type[AutoEncoderBase]

    @classmethod
    def get_domain_name(cls) -> str:
        return cls.__name__.split("_Domain")[0]

    @classmethod
    def create_solver(cls) -> AbstractScratchSolver:
        return cls.solver_type.init(cls.solver_config)

    @classmethod
    def get_multiprocess_batch_solver(
        cls, n_process: Optional[int] = None
    ) -> MultiProcessBatchTaskSolver:
        return MultiProcessBatchTaskSolver(
            cls.solver_type, cls.solver_config, cls.task_type, n_process=n_process
        )

    @classmethod
    def get_multiprocess_batch_sampler(
        cls, n_process: Optional[int] = None
    ) -> MultiProcessBatchTaskSampler:
        return MultiProcessBatchTaskSampler(n_process=n_process)

    @classmethod
    def get_distributed_batch_solver(cls, *args, **kwargs) -> DistributedBatchTaskSolver:
        return DistributedBatchTaskSolver(
            cls.solver_type, cls.solver_config, cls.task_type, *args, **kwargs
        )

    @classmethod
    def get_distributed_batch_sampler(cls, *args, **kwargs) -> DistributeBatchTaskSampler:
        return DistributeBatchTaskSampler(*args, **kwargs)


class FixedPR2MiniFridge_SQP(DomainProtocol):
    task_type = FixedPR2MiniFridgeTask
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(
        n_wp=60, n_max_call=5, motion_step_satisfaction="explicit", ineq_tighten_coef=0.0
    )
    auto_encoder_project_name = "PR2MiniFridge-AutoEncoder"
    auto_encoder_type = PixelAutoEncoder


class PR2MiniFridge_SQP(DomainProtocol):
    task_type = PR2MiniFridgeTask
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(
        n_wp=60, n_max_call=5, motion_step_satisfaction="explicit", ineq_tighten_coef=0.0
    )
    auto_encoder_project_name = "PR2MiniFridge-AutoEncoder"
    auto_encoder_type = PixelAutoEncoder


class PR2MiniFridge_RRT500(DomainProtocol):
    task_type = PR2MiniFridgeTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=500,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.1,
    )
    auto_encoder_project_name = "PR2MiniFridge-AutoEncoder"
    auto_encoder_type = PixelAutoEncoder


class PR2MiniFridge_RRT2000(DomainProtocol):
    task_type = PR2MiniFridgeTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=2000,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.1,
    )
    auto_encoder_project_name = "PR2MiniFridge-AutoEncoder"
    auto_encoder_type = PixelAutoEncoder


class PR2MiniFridge_RRT8000(DomainProtocol):
    task_type = PR2MiniFridgeTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=8000,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.1,
    )
    auto_encoder_project_name = "PR2MiniFridge-AutoEncoder"
    auto_encoder_type = PixelAutoEncoder


class PR2MiniFridgeVoxel_RRT500(DomainProtocol):
    task_type = PR2MiniFridgeVoxelTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=500,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.1,
    )
    auto_encoder_project_name = "PR2MiniFridge-VoxelAutoEncoder"
    auto_encoder_type = VoxelAutoEncoder


class PR2MiniFridgeVoxel_RRT2000(DomainProtocol):
    task_type = PR2MiniFridgeVoxelTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=2000,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.1,
    )
    auto_encoder_project_name = "PR2MiniFridge-VoxelAutoEncoder"
    auto_encoder_type = VoxelAutoEncoder


# class ClutteredFridgeRealistic_SQP(DomainProtocol):
#     task_type = TabletopClutteredFridgeReachingRealisticTask
#     solver_type = SQPBasedSolver
#     solver_config = SQPBasedSolverConfig(
#         n_wp=60, n_max_call=5, motion_step_satisfaction="explicit", ineq_tighten_coef=0.0
#     )
#     auto_encoder_project_name = "TabletopClutteredFridgeWorldWithRealisticContents-AutoEncoder"
#     auto_encoder_type = PixelAutoEncoder
#
#
# class ClutteredFridgeManyContents_SQP(DomainProtocol):
#     task_type = TabletopClutteredFridgeReachingManyContentsTask
#     solver_type = SQPBasedSolver
#     solver_config = SQPBasedSolverConfig(
#         n_wp=60, n_max_call=5, motion_step_satisfaction="explicit", ineq_tighten_coef=0.0
#     )
#     auto_encoder_project_name = "TODO"  # train this
#     auto_encoder_type = PixelAutoEncoder


# class ClutteredFridge_RRT250(DomainProtocol):
#     task_type = TabletopClutteredFridgeReachingTask
#     solver_type = OMPLSolver
#     solver_config = OMPLSolverConfig(
#         n_max_call=250,
#         n_max_satisfaction_trial=1,
#         expbased_planner_backend="ertconnect",
#         ertconnect_eps=0.1,
#     )
#     auto_encoder_project_name = "TabletopClutteredFridgeWorld-AutoEncoder"
#     auto_encoder_type = PixelAutoEncoder
#
#
# class ClutteredFridge_RRT500(DomainProtocol):
#     task_type = TabletopClutteredFridgeReachingTask
#     solver_type = OMPLSolver
#     solver_config = OMPLSolverConfig(
#         n_max_call=500,
#         n_max_satisfaction_trial=1,
#         expbased_planner_backend="ertconnect",
#         ertconnect_eps=0.1,
#     )
#     auto_encoder_project_name = "TabletopClutteredFridgeWorld-AutoEncoder"
#     auto_encoder_type = PixelAutoEncoder
#
#
# class ClutteredFridge_RRT1000(DomainProtocol):
#     task_type = TabletopClutteredFridgeReachingTask
#     solver_type = OMPLSolver
#     solver_config = OMPLSolverConfig(
#         n_max_call=1000,
#         n_max_satisfaction_trial=1,
#         expbased_planner_backend="ertconnect",
#         ertconnect_eps=0.1,
#     )
#     auto_encoder_project_name = "TabletopClutteredFridgeWorld-AutoEncoder"
#     auto_encoder_type = PixelAutoEncoder
#
#
# class ClutteredFridge_RRT2000(DomainProtocol):
#     task_type = TabletopClutteredFridgeReachingTask
#     solver_type = OMPLSolver
#     solver_config = OMPLSolverConfig(
#         n_max_call=2000,
#         n_max_satisfaction_trial=1,
#         expbased_planner_backend="ertconnect",
#         ertconnect_eps=0.1,
#     )
#     auto_encoder_project_name = "TabletopClutteredFridgeWorld-AutoEncoder"
#     auto_encoder_type = PixelAutoEncoder


class JSKFridge_SQP(DomainProtocol):
    task_type = JskFridgeReachingTask
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(
        n_wp=40, n_max_call=8, motion_step_satisfaction="explicit", ineq_tighten_coef=0.0
    )
    auto_encoder_project_name = "JskFridgeWorld-AutoEncoder"
    auto_encoder_type = PixelAutoEncoder


class JSKFridge_RRT2000(DomainProtocol):
    task_type = JskFridgeReachingTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=2000,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.1,
    )
    auto_encoder_project_name = "JskFridgeWorld-AutoEncoder"
    auto_encoder_type = PixelAutoEncoder


class JSKFridge_RRT5000(DomainProtocol):
    task_type = JskFridgeReachingTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=5000,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.1,
    )
    auto_encoder_project_name = "JskFridgeWorld-AutoEncoder"
    auto_encoder_type = PixelAutoEncoder


class JSKFridge_RRT10000(DomainProtocol):
    task_type = JskFridgeReachingTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=10000,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.1,
    )
    auto_encoder_project_name = "JskFridgeWorld-AutoEncoder"
    auto_encoder_type = PixelAutoEncoder


class JSKFridgeVertical_SQP(DomainProtocol):
    task_type = JskFridgeVerticalReachingTask
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(n_wp=40, n_max_call=8, motion_step_satisfaction="explicit")
    auto_encoder_project_name = "JskFridgeWorld-AutoEncoder"
    auto_encoder_type = PixelAutoEncoder


class JSKFridgeVertical_RRT2000(DomainProtocol):
    task_type = JskFridgeVerticalReachingTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=2000,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.1,
    )
    auto_encoder_project_name = "JskFridgeWorld-AutoEncoder"
    auto_encoder_type = PixelAutoEncoder


class JSKFridgeVertical2_SQP(DomainProtocol):
    task_type = JskFridgeVerticalReachingTask2
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(n_wp=40, n_max_call=8, motion_step_satisfaction="explicit")
    auto_encoder_project_name = "JskFridgeWorld2-AutoEncoder"
    auto_encoder_type = PixelAutoEncoder


class JSKFridgeVertical2_RRT2000(DomainProtocol):
    task_type = JskFridgeVerticalReachingTask2
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=2000,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.1,
    )
    auto_encoder_project_name = "JskFridgeWorld2-AutoEncoder"
    auto_encoder_type = PixelAutoEncoder


class HumanoidTableRarmReaching_SQP_Domain(DomainProtocol):
    task_type = HumanoidTableReachingTask
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(
        n_wp=40,
        n_max_call=5,
        motion_step_satisfaction="explicit",
        verbose=False,
        ctol_eq=1e-3,
        ctol_ineq=1e-3,
        ineq_tighten_coef=0.0,
    )
    auto_encoder_project_name = None
    auto_encoder_type = NullAutoEncoder


class HumanoidTableRarmReaching2_SQP_Domain(DomainProtocol):
    task_type = HumanoidTableReachingTask2
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(
        n_wp=40,
        n_max_call=5,
        motion_step_satisfaction="explicit",
        verbose=False,
        ctol_eq=1e-3,
        ctol_ineq=1e-3,
        ineq_tighten_coef=0.0,
    )
    auto_encoder_project_name = None
    auto_encoder_type = NullAutoEncoder


class HumanoidTableRarmReaching2_SQP10_Domain(DomainProtocol):
    task_type = HumanoidTableReachingTask2
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(
        n_wp=40,
        n_max_call=10,
        motion_step_satisfaction="explicit",
        verbose=False,
        ctol_eq=1e-3,
        ctol_ineq=1e-3,
        ineq_tighten_coef=0.0,
    )
    auto_encoder_project_name = None
    auto_encoder_type = NullAutoEncoder


class HumanoidTableRarmReaching3_SQP_Domain(DomainProtocol):
    task_type = HumanoidTableReachingTask3
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(
        n_wp=40,
        n_max_call=5,
        motion_step_satisfaction="explicit",
        verbose=False,
        ctol_eq=1e-3,
        ctol_ineq=1e-3,
        ineq_tighten_coef=0.0,
    )
    auto_encoder_project_name = None
    auto_encoder_type = NullAutoEncoder


class HumanoidTableClutteredRarmReaching_SQP_Domain(DomainProtocol):
    task_type = HumanoidTableClutteredReachingTask
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(
        n_wp=40,
        n_max_call=5,
        motion_step_satisfaction="explicit",
        verbose=False,
        ctol_eq=1e-3,
        ctol_ineq=1e-3,
        ineq_tighten_coef=0.0,
    )
    auto_encoder_project_name = "BelowTableClutteredWorld-AutoEncoder"
    auto_encoder_type = PixelAutoEncoder


class HumanoidTableClutteredRarmReaching2_SQP_Domain(DomainProtocol):
    task_type = HumanoidTableClutteredReachingTask2
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(
        n_wp=40,
        n_max_call=5,
        motion_step_satisfaction="explicit",
        verbose=False,
        ctol_eq=1e-3,
        ctol_ineq=1e-3,
        ineq_tighten_coef=0.0,
    )
    auto_encoder_project_name = "BelowTableClutteredWorld-AutoEncoder"
    auto_encoder_type = PixelAutoEncoder


class HumanoidTableClutteredRarmReaching2_SQP3_Domain(DomainProtocol):
    task_type = HumanoidTableClutteredReachingTask2
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(
        n_wp=40,
        n_max_call=3,
        motion_step_satisfaction="explicit",
        verbose=False,
        ctol_eq=1e-3,
        ctol_ineq=1e-3,
        ineq_tighten_coef=0.0,
    )
    auto_encoder_project_name = "BelowTableClutteredWorld-AutoEncoder"
    auto_encoder_type = PixelAutoEncoder


class ParametricMazeTask1D_SQP(DomainProtocol):
    task_type = ParametricMazeTask1D
    solver_type = DoubleIntegratorOptimizationSolver
    solver_config = DoubleIntegratorPlanningConfig(
        n_wp=150,
        n_max_call=10,
    )
    auto_encoder_project_name = None
    auto_encoder_type = NullAutoEncoder


class ParametricMazeTask2D_SQP(DomainProtocol):
    task_type = ParametricMazeTask2D
    solver_type = DoubleIntegratorOptimizationSolver
    solver_config = DoubleIntegratorPlanningConfig(
        n_wp=300,
        n_max_call=10,
    )
    auto_encoder_project_name = None
    auto_encoder_type = NullAutoEncoder


class ParametricMazeTask3D_SQP(DomainProtocol):
    task_type = ParametricMazeTask3D
    solver_type = DoubleIntegratorOptimizationSolver
    solver_config = DoubleIntegratorPlanningConfig(
        n_wp=450,
        n_max_call=10,
    )
    auto_encoder_project_name = None
    auto_encoder_type = NullAutoEncoder


class ParametricMazeTask4D_SQP(DomainProtocol):
    task_type = ParametricMazeTask4D
    solver_type = DoubleIntegratorOptimizationSolver
    solver_config = DoubleIntegratorPlanningConfig(
        n_wp=600,
        n_max_call=10,
    )
    auto_encoder_project_name = None
    auto_encoder_type = NullAutoEncoder


class ParametricMazeTask5D_SQP(DomainProtocol):
    task_type = ParametricMazeTask5D
    solver_type = DoubleIntegratorOptimizationSolver
    solver_config = DoubleIntegratorPlanningConfig(
        n_wp=750,
        n_max_call=10,
    )
    auto_encoder_project_name = None
    auto_encoder_type = NullAutoEncoder


class ParametricMazeTask4D_SQP(DomainProtocol):
    task_type = ParametricMazeSpecialTask
    solver_type = DoubleIntegratorOptimizationSolver
    solver_config = DoubleIntegratorPlanningConfig(
        n_wp=600,
        n_max_call=10,
    )
    auto_encoder_project_name = None
    auto_encoder_type = NullAutoEncoder


class DoubleIntegratorBubblyEmpty_SQP(DomainProtocol):
    task_type = BubblyEmptyMeshPointConnectTask
    solver_type = DoubleIntegratorOptimizationSolver
    solver_config = DoubleIntegratorPlanningConfig(
        n_wp=200,
        n_max_call=5,
    )
    auto_encoder_project_name = "BubblyWorldEmpty-AutoEncoder"
    auto_encoder_type = PixelAutoEncoder


class DoubleIntegratorBubblySimple_SQP(DomainProtocol):
    task_type = BubblySimpleMeshPointConnectTask
    solver_type = DoubleIntegratorOptimizationSolver
    solver_config = DoubleIntegratorPlanningConfig(
        n_wp=200,
        n_max_call=5,
    )
    auto_encoder_project_name = "BubblyWorldSimple-AutoEncoder"
    auto_encoder_type = PixelAutoEncoder


class DoubleIntegratorBubblyModerate_SQP(DomainProtocol):
    task_type = BubblyModerateMeshPointConnectTask
    solver_type = DoubleIntegratorOptimizationSolver
    solver_config = DoubleIntegratorPlanningConfig(
        n_wp=200,
        n_max_call=5,
    )
    auto_encoder_project_name = None
    auto_encoder_type = PixelAutoEncoder


class DoubleIntegratorBubblyComplex_SQP(DomainProtocol):
    task_type = BubblyComplexMeshPointConnectTask
    solver_type = DoubleIntegratorOptimizationSolver
    solver_config = DoubleIntegratorPlanningConfig(
        n_wp=200,
        n_max_call=5,
    )
    auto_encoder_project_name = "BubblyWorldComplex-AutoEncoder"
    auto_encoder_type = PixelAutoEncoder


class DummyDomain(DomainProtocol):
    task_type = DummyTask
    solver_type = DummySolver
    solver_config = DummyConfig(
        n_max_call=800, random_scale=0.25, random_force_failure_rate=0.0
    )  # somehow, if 500, classifier is not trained well probably due to the positive-negative sample inbalance
    auto_encoder_project_name = None
    auto_encoder_type = NullAutoEncoder


class DummyMeshDomain(DomainProtocol):
    task_type = DummyMeshTask
    solver_type = DummySolver
    solver_config = DummyConfig(
        n_max_call=800, random_scale=0.25, random_force_failure_rate=0.0
    )  # somehow, if 500, classifier is not trained well probably due to the positive-negative sample inbalance
    auto_encoder_project_name = None
    auto_encoder_type = PixelAutoEncoder


class ProbDummyDomain(DomainProtocol):
    task_type = ProbDummyTask
    solver_type = DummySolver
    solver_config = DummyConfig(
        n_max_call=200, random_scale=0.5, random_force_failure_rate=0.0
    )  # somehow, if 500, classifier is not trained well probably due to the positive-negative sample inbalance
    auto_encoder_project_name = None
    auto_encoder_type = NullAutoEncoder


def measure_time_per_call(domain: Type[DomainProtocol], n_sample: int = 10) -> float:
    solver = domain.create_solver()

    n_call_sum = 0
    elapsed_time_sum = 0.0
    for _ in tqdm.tqdm(range(n_sample)):
        task = domain.task_type.sample(1)
        problems = [p for p in task.export_problems()]
        solver.setup(problems[0])
        ts = time.time()
        res = solver.solve()
        elapsed_time_sum += time.time() - ts
        n_call_sum += res.n_call
    time_per_call_mean = elapsed_time_sum / n_call_sum
    return time_per_call_mean


def select_domain(domain_name: str) -> Type[DomainProtocol]:
    class DomainCollection(Enum):
        fixed_pr2_minifridge_sqp = FixedPR2MiniFridge_SQP
        pr2_minifridge_sqp = PR2MiniFridge_SQP
        pr2_minifridge_rrt500 = PR2MiniFridge_RRT500
        pr2_minifridge_rrt2000 = PR2MiniFridge_RRT2000
        pr2_minifridge_rrt8000 = PR2MiniFridge_RRT8000
        pr2_minifridge_voxel_rrt500 = PR2MiniFridgeVoxel_RRT500
        pr2_minifridge_voxel_rrt2000 = PR2MiniFridgeVoxel_RRT2000
        jsk_fridge_sqp = JSKFridge_SQP
        jsk_fridge_rrt2000 = JSKFridge_RRT2000
        jsk_fridge_rrt5000 = JSKFridge_RRT5000
        jsk_fridge_rrt10000 = JSKFridge_RRT10000
        jsk_fridge_vertical_sqp = JSKFridgeVertical_SQP
        jsk_fridge_vertical_rrt2000 = JSKFridgeVertical_RRT2000
        jsk_fridge_vertical2_sqp = JSKFridgeVertical2_SQP
        jsk_fridge_vertical2_rrt2000 = JSKFridgeVertical2_RRT2000
        humanoid_trr_sqp = HumanoidTableRarmReaching_SQP_Domain
        humanoid_trr2_sqp = HumanoidTableRarmReaching2_SQP_Domain
        humanoid_trr2_sqp10 = HumanoidTableRarmReaching2_SQP10_Domain
        humanoid_trr3_sqp = HumanoidTableRarmReaching3_SQP_Domain
        humanoid_tcrr_sqp = HumanoidTableClutteredRarmReaching_SQP_Domain
        humanoid_tcrr2_sqp = HumanoidTableClutteredRarmReaching2_SQP_Domain
        humanoid_tcrr2_sqp3 = HumanoidTableClutteredRarmReaching2_SQP3_Domain
        parametric_maze1d_sqp = ParametricMazeTask1D_SQP
        parametric_maze2d_sqp = ParametricMazeTask2D_SQP
        parametric_maze3d_sqp = ParametricMazeTask3D_SQP
        parametric_maze4d_sqp = ParametricMazeTask4D_SQP
        parametric_maze5d_sqp = ParametricMazeTask5D_SQP
        di_bubbly_moderate_sqp = DoubleIntegratorBubblyModerate_SQP
        di_bubbly_complex_sqp = DoubleIntegratorBubblyComplex_SQP
        di_bubbly_simple_sqp = DoubleIntegratorBubblySimple_SQP
        dummy = DummyDomain
        prob_dummy = ProbDummyDomain

    return DomainCollection[domain_name].value
