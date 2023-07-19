from typing import Dict, Optional

from rpbench.interface import (
    AbstractTaskSolver,
    DatadrivenTaskSolver,
    PlanningDataset,
    SkmpTaskSolver,
)
from rpbench.jaxon.below_table import HumanoidTableReachingTask
from rpbench.pr2.kivapod import KivapodEmptyReachingTask
from rpbench.pr2.tabletop import TabletopBoxRightArmReachingTask
from rpbench.two_dimensional.bubbly_world import BubblyComplexMeshPointConnectTask
from rpbench.two_dimensional.multiple_rooms import EightRoomsPlanningTask
from skmp.satisfy import SatisfactionConfig
from skmp.solver.myrrt_solver import MyRRTConfig, MyRRTConnectSolver
from skmp.solver.nlp_solver.memmo import NnMemmoSolver
from skmp.solver.nlp_solver.sqp_based_solver import SQPBasedSolverConfig
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig


class CompatibleSolvers:
    @classmethod
    def get_compatible_solvers(
        cls, task_name: str, dataset: Optional[PlanningDataset] = None
    ) -> Dict[str, AbstractTaskSolver]:
        method_name = "_{}".format(task_name)
        return getattr(cls, method_name)(dataset)

    @staticmethod
    def _EightRoomsPlanningTask(
        dataset: Optional[PlanningDataset],
    ) -> Dict[str, AbstractTaskSolver]:
        task_type = EightRoomsPlanningTask
        compat_solvers: Dict[str, AbstractTaskSolver] = {}
        ompl_config = OMPLSolverConfig(n_max_call=3000, n_max_satisfaction_trial=1)
        compat_solvers["rrtconnect"] = SkmpTaskSolver.init(OMPLSolver.init(ompl_config), task_type)
        return compat_solvers

    @staticmethod
    def _TabletopBoxRightArmReachingTask(
        dataset: Optional[PlanningDataset],
    ) -> Dict[str, AbstractTaskSolver]:
        task_type = TabletopBoxRightArmReachingTask
        compat_solvers: Dict[str, AbstractTaskSolver] = {}
        ompl_config = OMPLSolverConfig(n_max_call=5000, n_max_satisfaction_trial=20)
        compat_solvers["rrtconnect"] = SkmpTaskSolver.init(OMPLSolver.init(ompl_config), task_type)
        return compat_solvers

    @staticmethod
    def _TabletopBoxDualArmReachingTask(
        dataset: Optional[PlanningDataset],
    ) -> Dict[str, AbstractTaskSolver]:
        task_type = TabletopBoxRightArmReachingTask
        compat_solvers: Dict[str, AbstractTaskSolver] = {}
        ompl_config = OMPLSolverConfig(n_max_call=5000, n_max_satisfaction_trial=20)
        compat_solvers["rrtconnect"] = SkmpTaskSolver.init(OMPLSolver.init(ompl_config), task_type)
        return compat_solvers

    @staticmethod
    def _KivapodEmptyReachingTask(
        dataset: Optional[PlanningDataset],
    ) -> Dict[str, AbstractTaskSolver]:
        task_type = KivapodEmptyReachingTask

        compat_solvers: Dict[str, AbstractTaskSolver] = {}

        ompl_config = OMPLSolverConfig(n_max_call=3000, n_max_satisfaction_trial=30)
        sqp_config = SQPBasedSolverConfig(30, motion_step_satisfaction="explicit")

        # rrtconnect
        compat_solvers["rrtconnect"] = SkmpTaskSolver.init(OMPLSolver.init(ompl_config), task_type)

        # memmo
        assert dataset is not None
        compat_solvers["memmo_nn"] = DatadrivenTaskSolver.init(NnMemmoSolver, sqp_config, dataset)

        return compat_solvers

    @staticmethod
    def _HumanoidTableReachingTask(
        dataset: Optional[PlanningDataset],
    ) -> Dict[str, AbstractTaskSolver]:
        pass

        compat_solvers: Dict[str, AbstractTaskSolver] = {}

        myrrt_config = MyRRTConfig(3000, satisfaction_conf=SatisfactionConfig(n_max_eval=30))
        myrrt = MyRRTConnectSolver.init(myrrt_config)
        # myrrt_parallel4 = myrrt.as_parallel_solver(n_process=4)
        # myrrt_parallel8 = myrrt.as_parallel_solver(n_process=8)

        task_type = HumanoidTableReachingTask
        compat_solvers["rrtconnect"] = SkmpTaskSolver.init(myrrt, task_type)
        # compat_solvers["rrtconnect4"] = SkmpTaskSolver.init(myrrt_parallel4, task_type)
        # compat_solvers["rrtconnect8"] = SkmpTaskSolver.init(myrrt_parallel8, task_type)

        sqp_config = SQPBasedSolverConfig(30, motion_step_satisfaction="explicit")

        assert dataset is not None
        for n_experience in [50, 100, 200, 400, 800, 1600]:
            compat_solvers["memmo_nn{}".format(n_experience)] = DatadrivenTaskSolver.init(
                NnMemmoSolver,
                sqp_config,
                dataset,
                n_data_use=n_experience,
            )
        return compat_solvers

    @staticmethod
    def _BubblySimpleMeshPointConnectTask(
        dataset: Optional[PlanningDataset],
    ) -> Dict[str, AbstractTaskSolver]:
        task_type = BubblyComplexMeshPointConnectTask

        compat_solvers: Dict[str, AbstractTaskSolver] = {}

        # trajlib_dataset = PlanningDataset.load(task_type)

        rrt_config = OMPLSolverConfig(
            2000,
            n_max_satisfaction_trial=1,
            expbased_planner_backend="ertconnect",
            ertconnect_eps=0.5,
        )
        rrt = OMPLSolver.init(rrt_config)
        compat_solvers["rrtconnect"] = SkmpTaskSolver.init(rrt, task_type)

        # compat_solvers["ertconnect"] = DatadrivenTaskSolver.init(
        #     OMPLDataDrivenSolver, rrt_config, trajlib_dataset, n_data_use=300
        # )
        return compat_solvers