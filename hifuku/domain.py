import importlib.metadata
import time
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Optional, Protocol, Type

import packaging
import tqdm
from plainmp.ompl_solver import OMPLSolver as plainOMPLSolver
from plainmp.ompl_solver import OMPLSolverConfig as plainOMPLSolverConfig
from plainmp.ompl_solver import OMPLSolverResult as plainOMPLSolverResult
from plainmp.ompl_solver import TerminateState
from plainmp.problem import Problem
from rpbench.articulated.fetch.jail_insert import ConwayJailInsertTask, JailInsertTask

# from rpbench.articulated.fetch.tidyup_table import TidyupTableTask, TidyupTableTask2
from rpbench.articulated.jaxon.below_table import (
    HumanoidTableClutteredReachingTask,
    HumanoidTableClutteredReachingTask2,
    HumanoidTableReachingTask,
    HumanoidTableReachingTask2,
    HumanoidTableReachingTask3,
)
from rpbench.articulated.pr2.minifridge import (
    FixedPR2MiniFridgeTask,
    PR2MiniFridgeTask,
    PR2MiniFridgeVoxelTask,
)
from rpbench.articulated.pr2.thesis_jsk_table import (
    JskMessyTableTask,
    JskMessyTableTaskWithChair,
)
from rpbench.interface import TaskBase

from hifuku import is_plainmp_old

try:
    from rpbench.two_dimensional.bubbly_world import (
        BubblyComplexMeshPointConnectTask,
        BubblyEmptyMeshPointConnectTask,
        BubblyModerateMeshPointConnectTask,
        BubblySimpleMeshPointConnectTask,
        DoubleIntegratorOptimizationSolver,
        DoubleIntegratorPlanningConfig,
    )

    DISBMP_AVAILABLE = True
except ImportError:
    DISBMP_AVAILABLE = False

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
    ChannelSplitPixelAutoEncoder,
    NullAutoEncoder,
    PixelAutoEncoder,
    VoxelAutoEncoder,
)

# plainmp => skmp adapter


class PlainOMPLResultWrapper(plainOMPLSolverResult):
    @classmethod
    def abnormal(cls) -> "PlainOMPLResultWrapper":
        return cls(None, None, -1, TerminateState.FAIL_SATISFACTION)


@dataclass
class PlainOMPLSolverWrapper:
    solver: plainOMPLSolver
    problem: Optional[Problem]
    config: plainOMPLSolverConfig

    @classmethod
    def init(cls, config: plainOMPLSolverConfig):
        return cls(plainOMPLSolver(config), None, config)

    def setup(self, problem: Problem):
        self.problem = problem

    def solve(self, guess=None) -> PlainOMPLResultWrapper:
        return self._solve(guess)

    def _solve(self, guess=None) -> plainOMPLSolverResult:
        assert self.problem is not None
        ret = self.solver.solve(self.problem, guess)
        return ret

    def get_result_type(self) -> "Type[PlainOMPLResultWrapper]":
        return PlainOMPLResultWrapper


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


def is_plainmp_old():
    version_str = importlib.metadata.version("plainmp")
    return packaging.version.parse(version_str) < packaging.version.parse("0.0.8")


# class FetchTidyupTable(DomainProtocol):
#     task_type = TidyupTableTask
#     solver_type = PlainOMPLSolverWrapper
#     kwargs = {"n_max_call": 1000, "n_max_ik_trial": 1, "ertconnect_eps": 0.1}
#     if is_plainmp_old():
#         kwargs["expbased_planner_backend"] = "ertconnect"
#     solver_config = plainOMPLSolverConfig(**kwargs)
#     auto_encoder_project_name = "FetchTidyupTable-AutoEncoder"
#     auto_encoder_type = PixelAutoEncoder
#
#
# class FetchTidyupTable2(DomainProtocol):
#     task_type = TidyupTableTask2
#     solver_type = PlainOMPLSolverWrapper
#     kwargs = {"n_max_call": 1000, "n_max_ik_trial": 1, "ertconnect_eps": 0.1}
#     if is_plainmp_old():
#         kwargs["expbased_planner_backend"] = "ertconnect"
#     solver_config = plainOMPLSolverConfig(**kwargs)
#     auto_encoder_project_name = "FetchTidyupTable-AutoEncoder"
#     auto_encoder_type = PixelAutoEncoder


class FetchJailInsert(DomainProtocol):
    task_type = JailInsertTask
    solver_type = PlainOMPLSolverWrapper
    kwargs = {"n_max_call": 50000, "n_max_ik_trial": 1, "ertconnect_eps": 0.1, "timeout": 5.0}
    if is_plainmp_old():
        kwargs["expbased_planner_backend"] = "ertconnect"
    solver_config = plainOMPLSolverConfig(**kwargs)
    auto_encoder_project_name = "FetchJailInsert-AutoEncoder"
    auto_encoder_type = VoxelAutoEncoder


class FetchConwayJailInsert(DomainProtocol):
    task_type = ConwayJailInsertTask
    solver_type = PlainOMPLSolverWrapper
    kwargs = {"n_max_call": 50000, "n_max_ik_trial": 1, "ertconnect_eps": 0.1, "timeout": 5.0}
    if is_plainmp_old():
        kwargs["expbased_planner_backend"] = "ertconnect"
    solver_config = plainOMPLSolverConfig(**kwargs)
    auto_encoder_project_name = "FetchConwayJailInsert-AutoEncoder"
    auto_encoder_type = VoxelAutoEncoder


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


class Pr2ThesisJskTable(DomainProtocol):
    task_type = JskMessyTableTask
    solver_type = PlainOMPLSolverWrapper
    solver_config = plainOMPLSolverConfig(
        n_max_call=10000,
        n_max_ik_trial=1,
        ertconnect_eps=0.1,
    )
    auto_encoder_project_name = "Pr2ThesisJskTable-AutoEncoder"
    auto_encoder_type = PixelAutoEncoder


class Pr2ThesisJskTable2(DomainProtocol):
    task_type = JskMessyTableTaskWithChair
    solver_type = PlainOMPLSolverWrapper
    solver_config = plainOMPLSolverConfig(
        n_max_call=10000,
        n_max_ik_trial=1,
        ertconnect_eps=0.1,
    )
    auto_encoder_project_name = "Pr2ThesisJskTable2-AutoEncoder"
    auto_encoder_type = ChannelSplitPixelAutoEncoder


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


if DISBMP_AVAILABLE:

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

else:

    class DoubleIntegratorBubblyEmpty_SQP(DomainProtocol):
        ...

    class DoubleIntegratorBubblySimple_SQP(DomainProtocol):
        ...

    class DoubleIntegratorBubblyModerate_SQP(DomainProtocol):
        ...

    class DoubleIntegratorBubblyComplex_SQP(DomainProtocol):
        ...


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
        fetch_jail = FetchJailInsert
        fetch_cjail = FetchConwayJailInsert
        # fetch_tidyup = FetchTidyupTable
        # fetch_tidyup2 = FetchTidyupTable2
        fixed_pr2_minifridge_sqp = FixedPR2MiniFridge_SQP
        pr2_minifridge_sqp = PR2MiniFridge_SQP
        pr2_minifridge_rrt500 = PR2MiniFridge_RRT500
        pr2_minifridge_rrt2000 = PR2MiniFridge_RRT2000
        pr2_minifridge_rrt8000 = PR2MiniFridge_RRT8000
        pr2_minifridge_voxel_rrt500 = PR2MiniFridgeVoxel_RRT500
        pr2_minifridge_voxel_rrt2000 = PR2MiniFridgeVoxel_RRT2000
        pr2_thesis_jsk_table = Pr2ThesisJskTable
        pr2_thesis_jsk_table2 = Pr2ThesisJskTable2
        jsk_fridge = JSKFridge
        jsk_fridge_grasping = JSKFridgeGrasping
        humanoid_trr_sqp = HumanoidTableRarmReaching_SQP_Domain
        humanoid_trr2_sqp = HumanoidTableRarmReaching2_SQP_Domain
        humanoid_trr2_sqp10 = HumanoidTableRarmReaching2_SQP10_Domain
        humanoid_trr3_sqp = HumanoidTableRarmReaching3_SQP_Domain
        humanoid_tcrr_sqp = HumanoidTableClutteredRarmReaching_SQP_Domain
        humanoid_tcrr2_sqp = HumanoidTableClutteredRarmReaching2_SQP_Domain
        humanoid_tcrr2_sqp3 = HumanoidTableClutteredRarmReaching2_SQP3_Domain
        di_bubbly_moderate_sqp = DoubleIntegratorBubblyModerate_SQP
        di_bubbly_complex_sqp = DoubleIntegratorBubblyComplex_SQP
        di_bubbly_simple_sqp = DoubleIntegratorBubblySimple_SQP
        dummy = DummyDomain
        prob_dummy = ProbDummyDomain

    return DomainCollection[domain_name].value
