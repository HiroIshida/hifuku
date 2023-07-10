import time
from enum import Enum
from typing import ClassVar, Optional, Protocol, Type

import tqdm
from skmp.solver.interface import AbstractScratchSolver, ConfigProtocol
from skmp.solver.nlp_solver.sqp_based_solver import SQPBasedSolver, SQPBasedSolverConfig
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig

from hifuku.datagen.batch_sampler import (
    DistributeBatchProblemSampler,
    MultiProcessBatchProblemSampler,
)
from hifuku.datagen.batch_solver import (
    DistributedBatchProblemSolver,
    MultiProcessBatchProblemSolver,
)
from hifuku.rpbench_wrap import (
    BubblySimpleMeshPointConnectTask,
    BubblySimplePointConnectTask,
    EightRoomsPlanningTask,
    HumanoidTableReachingTask,
    KivapodEmptyReachingTask,
    MazeSolvingTask,
    PicklableTaskBase,
    RingObstacleFreeBlockedPlanningTask,
    RingObstacleFreePlanningTask,
    TabletopBoxDualArmReachingTask,
    TabletopOvenDualArmReachingTask,
    TabletopOvenRightArmReachingTask,
)


class DomainProtocol(Protocol):
    task_type: ClassVar[Type[PicklableTaskBase]]
    solver_type: ClassVar[Type[AbstractScratchSolver]]
    solver_config: ClassVar[ConfigProtocol]
    auto_encoder_project_name: ClassVar[Optional[str]]

    @classmethod
    def get_domain_name(cls) -> str:
        return cls.__name__.split("_Domain")[0]

    @classmethod
    def create_solver(cls) -> AbstractScratchSolver:
        return cls.solver_type.init(cls.solver_config)

    @classmethod
    def get_multiprocess_batch_solver(
        cls, n_process: Optional[int] = None
    ) -> MultiProcessBatchProblemSolver:
        return MultiProcessBatchProblemSolver(
            cls.solver_type, cls.solver_config, n_process=n_process
        )

    @classmethod
    def get_multiprocess_batch_sampler(
        cls, n_process: Optional[int] = None
    ) -> MultiProcessBatchProblemSampler:
        return MultiProcessBatchProblemSampler(n_process=n_process)

    @classmethod
    def get_distributed_batch_solver(cls, *args, **kwargs) -> DistributedBatchProblemSolver:
        return DistributedBatchProblemSolver(cls.solver_type, cls.solver_config, *args, **kwargs)

    @classmethod
    def get_distributed_batch_sampler(cls, *args, **kwargs) -> DistributeBatchProblemSampler:
        return DistributeBatchProblemSampler(*args, **kwargs)


class TORR_RRT_Domain(DomainProtocol):
    task_type = TabletopOvenRightArmReachingTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=3000,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.5,
    )
    auto_encoder_project_name = "hifuku-TabletopOvenWorldWrap"


class TORR_SQP_Domain(DomainProtocol):
    task_type = TabletopOvenRightArmReachingTask
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(n_wp=50, n_max_call=5, motion_step_satisfaction="explicit")
    auto_encoder_project_name = "hifuku-TabletopOvenWorldWrap"


class TODR_SQP_Domain(DomainProtocol):
    task_type = TabletopOvenDualArmReachingTask
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(n_wp=50, n_max_call=5, motion_step_satisfaction="explicit")
    auto_encoder_project_name = "hifuku-TabletopOvenWorldWrap"


class TBDR_SQP_Domain(DomainProtocol):
    task_type = TabletopBoxDualArmReachingTask
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(
        n_wp=60, n_max_call=5, motion_step_satisfaction="explicit", ineq_tighten_coef=0.0
    )
    auto_encoder_project_name = "hifuku-TabletopBoxWorldWrap"


class TBDR_RRT_Domain(DomainProtocol):
    task_type = TabletopBoxDualArmReachingTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=1000,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.5,
    )
    auto_encoder_project_name = "hifuku-TabletopBoxWorldWrap"


class TBRR_RRT_Domain(DomainProtocol):
    task_type = TabletopBoxRightArmReachingTask
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(
        n_wp=60, n_max_call=5, motion_step_satisfaction="explicit", ineq_tighten_coef=0.0
    )
    auto_encoder_project_name = "hifuku-TabletopBoxWorldWrap"


class Kivapod_Empty_RRT_Domain(DomainProtocol):
    task_type = KivapodEmptyReachingTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=3000,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.5,
    )
    auto_encoder_project_name = None


class Maze_RRT_Domain(DomainProtocol):
    task_type = MazeSolvingTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=3000,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.5,
    )
    auto_encoder_project_name = None


class RingObstacleFree_RRT_Domain(DomainProtocol):
    task_type = RingObstacleFreePlanningTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=100,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.1,
    )
    auto_encoder_project_name = None


class RingObstacleFreeBlocked_RRT_Domain(DomainProtocol):
    task_type = RingObstacleFreeBlockedPlanningTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=100,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.1,
    )
    auto_encoder_project_name = None


class EightRooms_SQP_Domain(DomainProtocol):
    task_type = EightRoomsPlanningTask
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(
        n_wp=20, n_max_call=10, motion_step_satisfaction="explicit"
    )
    auto_encoder_project_name = None


class EightRooms_RRT_Domain(DomainProtocol):
    task_type = EightRoomsPlanningTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        600,
        1,
        ertconnect_eps=0.5,
        algorithm_range=0.1,
        simplify=False,
        expbased_planner_backend="ertconnect",
    )
    auto_encoder_project_name = None


class HumanoidTableRarmReaching_SQP_Domain(DomainProtocol):
    task_type = HumanoidTableReachingTask
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(
        n_wp=40,
        n_max_call=8,
        motion_step_satisfaction="explicit",
        verbose=False,
        ctol_eq=1e-3,
        ctol_ineq=1e-3,
        ineq_tighten_coef=0.0,
    )
    auto_encoder_project_name = None


class BubblySimpleMeshPointConnecting_SQP_Domain(DomainProtocol):
    task_type = BubblySimpleMeshPointConnectTask
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(
        n_wp=20,
        n_max_call=20,
        motion_step_satisfaction="implicit",
        verbose=False,
    )
    auto_encoder_project_name = "BubblyWorldSimple-AutoEncoder"


class BubblySimpleMeshPointConnecting_RRT_Domain(DomainProtocol):
    task_type = BubblySimpleMeshPointConnectTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        200, 1, expbased_planner_backend="ertconnect", ertconnect_eps=0.3
    )
    auto_encoder_project_name = "BubblyWorldSimple-AutoEncoder"


class BubblySimplePointConnecting_RRT_Domain(DomainProtocol):
    task_type = BubblySimplePointConnectTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        200, 1, expbased_planner_backend="ertconnect", ertconnect_eps=0.3
    )
    auto_encoder_project_name = "BubblyWorldSimple-AutoEncoder"


def measure_time_per_call(domain: Type[DomainProtocol], n_sample: int = 10) -> float:
    solver = domain.create_solver()

    n_call_sum = 0
    elapsed_time_sum = 0.0
    for _ in tqdm.tqdm(range(n_sample)):
        task = domain.task_type.sample(1)
        problem = task.export_problems()[0]
        solver.setup(problem)
        ts = time.time()
        res = solver.solve()
        elapsed_time_sum += time.time() - ts
        n_call_sum += res.n_call
    time_per_call_mean = elapsed_time_sum / n_call_sum
    return time_per_call_mean


def select_domain(domain_name: str) -> Type[DomainProtocol]:
    class DomainCollection(Enum):
        torr_sqp = TORR_SQP_Domain
        torr_rrt = TORR_RRT_Domain
        todr_sqp = TODR_SQP_Domain
        tbdr_sqp = TBDR_SQP_Domain
        tbdr_rrt = TBDR_RRT_Domain
        tbrr_sqp = TBRR_SQP_Domain
        kivapod_empty_rrt = Kivapod_Empty_RRT_Domain
        ring_rrt = RingObstacleFree_RRT_Domain
        ring_blocked_rrt = RingObstacleFreeBlocked_RRT_Domain
        eight_rooms_sqp = EightRooms_SQP_Domain
        eight_rooms_rrt = EightRooms_RRT_Domain
        humanoid_trr_sqp = HumanoidTableRarmReaching_SQP_Domain
        bubbly_simple_mesh_sqp = BubblySimpleMeshPointConnecting_SQP_Domain
        bubbly_simple_mesh_rrt = BubblySimpleMeshPointConnecting_RRT_Domain
        bubbly_simple_rrt = BubblySimplePointConnecting_RRT_Domain

    return DomainCollection[domain_name].value
