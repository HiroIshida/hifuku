import time
from enum import Enum
from typing import ClassVar, Optional, Protocol, Type

import tqdm
from rpbench.interface import SamplableBase
from skmp.solver.interface import AbstractScratchSolver, ConfigProtocol
from skmp.solver.nlp_solver import SQPBasedSolver, SQPBasedSolverConfig
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
    EightRoomsPlanningTask,
    HumanoidTableReachingTask,
    KivapodEmptyReachingTask,
    MazeSolvingTask,
    PicklableTaskBase,
    RingObstacleFreeBlockedPlanningTask,
    RingObstacleFreePlanningTask,
    TabletopBoxDualArmReachingTask,
    TabletopBoxRightArmReachingTask,
    TabletopBoxWorldWrap,
)


class DomainProtocol(Protocol):
    task_type: ClassVar[Type[PicklableTaskBase]]
    solver_type: ClassVar[Type[AbstractScratchSolver]]
    solver_config: ClassVar[ConfigProtocol]
    mesh_sampler_type: ClassVar[Optional[Type[SamplableBase]]]

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


class TBRR_RRT_Domain(DomainProtocol):
    task_type = TabletopBoxRightArmReachingTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=3000,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.5,
    )
    mesh_sampler_type = TabletopBoxWorldWrap


class TBRR_SQP_Domain(DomainProtocol):
    task_type = TabletopBoxRightArmReachingTask
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(n_wp=50, n_max_call=5, motion_step_satisfaction="explicit")
    mesh_sampler_type = TabletopBoxWorldWrap


class TBDR_SQP_Domain(DomainProtocol):
    task_type = TabletopBoxDualArmReachingTask
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(n_wp=50, n_max_call=5, motion_step_satisfaction="explicit")
    mesh_sampler_type = TabletopBoxWorldWrap


class Kivapod_Empty_RRT_Domain(DomainProtocol):
    task_type = KivapodEmptyReachingTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=3000,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.5,
    )
    mesh_sampler_type = None


class Maze_RRT_Domain(DomainProtocol):
    task_type = MazeSolvingTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=3000,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.5,
    )
    mesh_sampler_type = None


class RingObstacleFree_RRT_Domain(DomainProtocol):
    task_type = RingObstacleFreePlanningTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=100,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.1,
    )
    mesh_sampler_type = None


class RingObstacleFreeBlocked_RRT_Domain(DomainProtocol):
    task_type = RingObstacleFreeBlockedPlanningTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(
        n_max_call=100,
        n_max_satisfaction_trial=1,
        expbased_planner_backend="ertconnect",
        ertconnect_eps=0.1,
    )
    mesh_sampler_type = None


class EightRooms_SQP_Domain(DomainProtocol):
    task_type = EightRoomsPlanningTask
    solver_type = SQPBasedSolver
    solver_config = SQPBasedSolverConfig(
        n_wp=60, n_max_call=10, motion_step_satisfaction="explicit"
    )
    mesh_sampler_type = None


class EightRooms_Lightning_Domain(DomainProtocol):
    task_type = EightRoomsPlanningTask
    solver_type = OMPLSolver
    solver_config = OMPLSolverConfig(200, 1, simplify=False, expbased_planner_backend="lightning")
    mesh_sampler_type = None


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
    mesh_sampler_type = None


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
        tbrr_sqp = TBRR_SQP_Domain
        tbrr_rrt = TBRR_RRT_Domain
        tbdr_sqp = TBDR_SQP_Domain
        kivapod_empty_rrt = Kivapod_Empty_RRT_Domain
        ring_rrt = RingObstacleFree_RRT_Domain
        ring_blocked_rrt = RingObstacleFreeBlocked_RRT_Domain
        eight_rooms_sqp = EightRooms_SQP_Domain
        eight_rooms_lt = EightRooms_Lightning_Domain
        humanoid_trr_sqp = HumanoidTableRarmReaching_SQP_Domain

    return DomainCollection[domain_name].value
