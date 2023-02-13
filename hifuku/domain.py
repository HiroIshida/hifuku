from abc import ABC, abstractmethod
from typing import Generic, Optional, Type

from rpbench.interface import SamplableBase
from skmp.solver.interface import AbstractScratchSolver, ConfigT, ResultT
from skmp.solver.nlp_solver import (
    SQPBasedSolver,
    SQPBasedSolverConfig,
    SQPBasedSolverResult,
)
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig, OMPLSolverResult

from hifuku.datagen import (
    DistributeBatchProblemSampler,
    DistributedBatchProblemSolver,
    MultiProcessBatchProblemSampler,
    MultiProcessBatchProblemSolver,
)
from hifuku.pool import ProblemT
from hifuku.rpbench_wrap import (
    MazeSolvingTask,
    TabletopBoxDualArmReachingTask,
    TabletopBoxRightArmReachingTask,
    TabletopBoxWorldWrap,
)


class DomainProvider(ABC, Generic[ProblemT, ConfigT, ResultT]):
    """*stateless* Domain informatino Provider
    where domain is composed of task_type, solver, solver_config
    """

    @classmethod
    @abstractmethod
    def get_task_type(cls) -> Type[ProblemT]:
        ...

    @classmethod
    @abstractmethod
    def get_solver_type(cls) -> Type[AbstractScratchSolver[ConfigT, ResultT]]:
        ...

    @classmethod
    @abstractmethod
    def get_solver_config(cls) -> ConfigT:
        ...

    @classmethod
    @abstractmethod
    def get_compat_mesh_sampler_type(
        cls,
    ) -> Optional[Type[SamplableBase]]:  # TODO: ok to use it directly from rpbench?
        ...

    @classmethod
    def get_multiprocess_batch_solver(
        cls, n_process: Optional[int] = None
    ) -> MultiProcessBatchProblemSolver[ConfigT, ResultT]:
        return MultiProcessBatchProblemSolver(
            cls.get_solver_type(), cls.get_solver_config(), n_process=n_process
        )

    @classmethod
    def get_multiprocess_batch_sampler(
        cls, n_process: Optional[int] = None
    ) -> MultiProcessBatchProblemSampler[ProblemT]:
        return MultiProcessBatchProblemSampler[ProblemT](n_process=n_process)

    @classmethod
    def get_distributed_batch_solver(
        cls, *args, **kwargs
    ) -> DistributedBatchProblemSolver[ConfigT, ResultT]:
        return DistributedBatchProblemSolver(
            cls.get_solver_type(), cls.get_solver_config(), *args, **kwargs
        )

    @classmethod
    def get_distributed_batch_sampler(
        cls, *args, **kwargs
    ) -> DistributeBatchProblemSampler[ProblemT]:
        return DistributeBatchProblemSampler[ProblemT](*args, **kwargs)

    @classmethod
    def get_domain_name(cls) -> str:
        return cls.__name__.split("_DomainProvider")[0]


class TBRR_RRT_DomainProvider(
    DomainProvider[TabletopBoxRightArmReachingTask, OMPLSolverConfig, OMPLSolverResult]
):
    @classmethod
    def get_task_type(cls) -> Type[TabletopBoxRightArmReachingTask]:
        return TabletopBoxRightArmReachingTask

    @classmethod
    def get_solver_type(
        cls,
    ) -> Type[AbstractScratchSolver[OMPLSolverConfig, OMPLSolverResult]]:
        return OMPLSolver

    @classmethod
    def get_solver_config(cls) -> OMPLSolverConfig:
        return OMPLSolverConfig(
            n_max_call=3000,
            n_max_satisfaction_trial=1,
            expbased_planner_backend="ertconnect",
            ertconnect_eps=0.5,
        )

    @classmethod
    @abstractmethod
    def get_compat_mesh_sampler_type(cls) -> Optional[Type[SamplableBase]]:
        return TabletopBoxWorldWrap


class TBRR_SQP_DomainProvider(
    DomainProvider[TabletopBoxRightArmReachingTask, SQPBasedSolverConfig, SQPBasedSolverResult]
):
    @classmethod
    def get_task_type(cls) -> Type[TabletopBoxRightArmReachingTask]:
        return TabletopBoxRightArmReachingTask

    @classmethod
    def get_solver_type(
        cls,
    ) -> Type[AbstractScratchSolver[SQPBasedSolverConfig, SQPBasedSolverResult]]:
        return SQPBasedSolver

    @classmethod
    def get_solver_config(cls) -> SQPBasedSolverConfig:
        return SQPBasedSolverConfig(n_wp=50, n_max_call=5, motion_step_satisfaction="explicit")

    @classmethod
    @abstractmethod
    def get_compat_mesh_sampler_type(cls) -> Optional[Type[SamplableBase]]:
        return TabletopBoxWorldWrap


class TBDR_SQP_DomainProvider(
    DomainProvider[TabletopBoxDualArmReachingTask, SQPBasedSolverConfig, SQPBasedSolverResult]
):
    @classmethod
    def get_task_type(cls) -> Type[TabletopBoxDualArmReachingTask]:
        return TabletopBoxDualArmReachingTask

    @classmethod
    def get_solver_type(
        cls,
    ) -> Type[AbstractScratchSolver[SQPBasedSolverConfig, SQPBasedSolverResult]]:
        return SQPBasedSolver

    @classmethod
    def get_solver_config(cls) -> SQPBasedSolverConfig:
        return SQPBasedSolverConfig(n_wp=50, n_max_call=5, motion_step_satisfaction="explicit")

    @classmethod
    @abstractmethod
    def get_compat_mesh_sampler_type(cls) -> Optional[Type[SamplableBase]]:
        return TabletopBoxWorldWrap


class Maze_RRT_DomainProvider(DomainProvider[MazeSolvingTask, OMPLSolverConfig, OMPLSolverResult]):
    @classmethod
    def get_task_type(cls) -> Type[MazeSolvingTask]:
        return MazeSolvingTask

    @classmethod
    def get_solver_type(
        cls,
    ) -> Type[AbstractScratchSolver[OMPLSolverConfig, OMPLSolverResult]]:
        return OMPLSolver

    @classmethod
    def get_solver_config(cls) -> OMPLSolverConfig:
        return OMPLSolverConfig(
            n_max_call=3000,
            n_max_satisfaction_trial=1,
            expbased_planner_backend="ertconnect",
            ertconnect_eps=0.5,
        )

    @classmethod
    @abstractmethod
    def get_compat_mesh_sampler_type(cls) -> Optional[Type[SamplableBase]]:
        return None
