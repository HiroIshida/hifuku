import copy
import logging
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import (
    Callable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import skrobot
import torch
from skplan.robot.pr2 import PR2Paramter
from skplan.solver.constraint import (
    ConstraintSatisfactionFail,
    ObstacleAvoidanceConstraint,
    PoseConstraint,
    TrajectoryEqualityConstraint,
    TrajectoryInequalityConstraint,
    batch_sample_from_manifold,
)
from skplan.solver.inverse_kinematics import IKConfig, IKResult, InverseKinematicsSolver
from skplan.solver.optimization import OsqpSqpPlanner
from skplan.solver.rrt import BidirectionalRRT, RRTConfig, StartConstraintRRT
from skplan.space import ConfigurationSpace, TaskSpace
from skplan.trajectory import Trajectory
from skplan.viewer.skrobot_viewer import get_robot_config, set_robot_config
from skrobot.coordinates import Coordinates
from skrobot.model import Axis
from skrobot.model.link import Link
from skrobot.model.primitives import Box
from skrobot.sdf import SignedDistanceFunction, UnionSDF
from skrobot.viewers import TrimeshSceneViewer
from voxbloxpy.core import EsdfMap, Grid, GridSDF, IntegratorType

from hifuku.llazy.dataset import (
    DatasetIterator,
    LazyDecomplessDataset,
    PicklableChunkBase,
)
from hifuku.pool import FixedProblemPool
from hifuku.sdf import create_union_sdf
from hifuku.threedim.camera import RayMarchingConfig, create_synthetic_esdf
from hifuku.threedim.robot import get_pr2_kinect_camera, setup_kinmaps, setup_pr2
from hifuku.threedim.utils import skcoords_to_pose_vec
from hifuku.types import ProblemInterface, ResultProtocol

logger = logging.getLogger(__name__)


@dataclass
class TableTopWorld:
    table: Box
    obstacles: List[Link]
    box_center: Coordinates
    box_d: float
    box_w: float
    box_h: float
    box_t: float

    def get_union_sdf(self) -> UnionSDF:
        lst = [self.table.sdf]
        for obstacle in self.obstacles:
            lst.append(obstacle.sdf)
        return UnionSDF(lst)

    def get_grid(
        self, grid_sizes: Tuple[int, int, int] = (56, 56, 28), mesh_height: float = 0.5
    ) -> Grid:
        depth, width, height = self.table._extents
        lb = np.array([-0.5 * depth, -0.5 * width, 0.5 * height - 0.1])
        ub = np.array([+0.5 * depth, +0.5 * width, 0.5 * height + mesh_height])
        lb = self.table.transform_vector(lb)
        ub = self.table.transform_vector(ub)
        return Grid(lb, ub, grid_sizes)

    def sample_pose(self, standard: bool = False) -> Coordinates:
        table = self.table
        table_depth, table_width, table_height = table._extents

        co = self.box_center.copy_worldcoords()
        if standard:
            d_trans = 0.0
            w_trans = 0.0
            h_trans = 0.5 * self.box_h
            theta = 0.0
        else:
            margin = 0.03
            box_dt = self.box_d - 2 * (self.box_t + margin)
            box_wt = self.box_w - 2 * (self.box_t + margin)
            box_ht = self.box_h - 2 * (self.box_t + margin)
            d_trans = -0.5 * box_dt + np.random.rand() * box_dt
            w_trans = -0.5 * box_wt + np.random.rand() * box_wt
            h_trans = self.box_t + margin + np.random.rand() * box_ht
            theta = -np.deg2rad(45) + np.random.rand() * np.deg2rad(90)

        co.translate([d_trans, w_trans, h_trans])
        co.rotate(theta, "z")

        points = np.expand_dims(co.worldpos(), axis=0)
        sdf = self.get_union_sdf()

        sd_val = sdf(points)[0]
        assert sd_val > -0.0001
        return co

    @classmethod
    def sample(cls, standard: bool = False) -> "TableTopWorld":
        table = cls.create_standard_table()
        table_depth, table_width, table_height = table._extents
        x = np.random.rand() * 0.2
        y = -0.2 + np.random.rand() * 0.4
        z = 0.0
        table.translate([x, y, z])

        table_tip = table.copy_worldcoords()
        table_tip.translate([-table_depth * 0.5, -table_width * 0.5, +0.5 * table_height])

        obstacles = []

        color = np.array([255, 100, 0, 200])

        # box
        d = 0.2 + np.random.rand() * 0.3
        w = 0.3 + np.random.rand() * 0.3
        h = 0.2 + np.random.rand() * 0.3
        t = 0.03

        if standard:
            box_center = table.copy_worldcoords()
            box_center.translate([0, 0, 0.5 * table_height])
        else:
            box_center = table_tip.copy_worldcoords()
            box_center.translate([0.5 * d, 0.5 * w, 0.0])
            pos_from_tip = np.array([table_depth - d, table_width - w, 0]) * np.random.rand(3)
            box_center.translate(pos_from_tip)

        lower_plate = Box([d, w, t], with_sdf=True, face_colors=color)
        lower_plate.newcoords(box_center.copy_worldcoords())
        lower_plate.translate([0, 0, 0.5 * t])
        obstacles.append(lower_plate)

        upper_plate = Box([d, w, t], with_sdf=True, face_colors=color)
        upper_plate.newcoords(box_center.copy_worldcoords())
        upper_plate.translate([0, 0, h - 0.5 * t])
        obstacles.append(upper_plate)

        left_plate = Box([d, t, h], with_sdf=True, face_colors=color)
        left_plate.newcoords(box_center.copy_worldcoords())
        left_plate.translate([0, 0.5 * w - 0.5 * t, 0.5 * h])
        obstacles.append(left_plate)

        right_plate = Box([d, t, h], with_sdf=True, face_colors=color)
        right_plate.newcoords(box_center.copy_worldcoords())
        right_plate.translate([0, -0.5 * w + 0.5 * t, 0.5 * h])
        obstacles.append(right_plate)

        opposite_plate = Box([t, w, h], with_sdf=True, face_colors=color)
        opposite_plate.newcoords(box_center.copy_worldcoords())
        opposite_plate.translate([0.5 * d - 0.5 * t, 0.0, 0.5 * h])
        obstacles.append(opposite_plate)

        return cls(table, obstacles, box_center, d, w, h, t)

    @staticmethod
    def create_standard_table() -> Box:
        # create jsk-lab 73b2 table
        table_depth = 0.5
        table_width = 0.75
        table_height = 0.7
        pos = [0.5 + table_depth * 0.5, 0.0, table_height * 0.5]
        table = Box(extents=[table_depth, table_width, table_height], pos=pos, with_sdf=True)
        return table


TableTopProblemT = TypeVar("TableTopProblemT", bound="_TabletopProblem")


@dataclass
class _TabletopProblem(PicklableChunkBase, ProblemInterface):
    world: TableTopWorld
    target_pose_list: List[Coordinates]

    # The reason for this aux cache is to enable to set cache from outside
    # Primary usecase for this cache is in mesh generation problems.
    _aux_gridsdf_cache: Optional[GridSDF] = None

    def cast_to(self, problem_type: Type[TableTopProblemT]) -> TableTopProblemT:
        return problem_type(self.world, self.target_pose_list, self._aux_gridsdf_cache)

    def to_tensors(self) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        raise NotImplementedError

    @abstractmethod
    def create_gridsdf(self, grid: Grid, sdf: SignedDistanceFunction) -> GridSDF:
        ...

    @cached_property
    def grid_sdf(self) -> GridSDF:
        # NOTE: you may think, removing cached_property and rather, put the cache
        # into _aux_gridsdf_caceh. However, this is problematic if we sending and
        # back the problems through network, because the size of the instance will
        # be huge. Cached property is free from this problem, as when pickling the
        # problem, the internal state inside cached_property will be removed.
        if self._aux_gridsdf_cache is not None:
            return self._aux_gridsdf_cache
        else:
            grid = self.world.get_grid()
            exact_obstacle_sdf = UnionSDF([obs.sdf for obs in self.world.obstacles])
            gridsdf = self.create_gridsdf(grid, exact_obstacle_sdf)
        return gridsdf

    def get_sdf(self) -> Callable[[np.ndarray], np.ndarray]:
        sdf = create_union_sdf((self.grid_sdf, self.world.table.sdf))  # type: ignore
        return sdf

    def get_mesh(self) -> np.ndarray:
        grid_sdf = self.grid_sdf
        return grid_sdf.values.reshape(grid_sdf.grid.sizes)

    def add_elements_to_viewer(self, viewer: TrimeshSceneViewer) -> None:
        viewer.add(self.world.table)
        for obs in self.world.obstacles:
            viewer.add(obs)
        for pose in self.target_pose_list:
            axis = Axis.from_coords(pose)
            viewer.add(axis)

    def get_descriptions(self) -> List[np.ndarray]:
        table_pose = skcoords_to_pose_vec(self.world.table.worldcoords())
        # description vector is composed of 6 + 6 dimension
        return [
            np.hstack([skcoords_to_pose_vec(pose), table_pose]) for pose in self.target_pose_list
        ]

    def n_problem(self) -> int:
        return len(self.target_pose_list)

    @classmethod
    def get_description_dim(cls) -> int:
        return 12

    @classmethod
    def create_standard(cls: Type[TableTopProblemT]) -> TableTopProblemT:
        # TODO: move to sample(standard=True) ??
        world = TableTopWorld.sample(standard=True)
        pose = world.sample_pose(standard=True)
        return cls(world, [pose])

    # fmt: off
    @classmethod
    @overload
    def sample(cls: Type[TableTopProblemT], n_pose: int, predicate: Callable[[TableTopProblemT], bool], max_trial_factor: int = ...) -> Optional[TableTopProblemT]: ...  # noqa

    @classmethod
    @overload
    def sample(cls: Type[TableTopProblemT], n_pose: int, predicate: None, max_trial_factor: int = ...) -> TableTopProblemT: ...  # noqa

    @classmethod
    @overload
    def sample(cls: Type[TableTopProblemT], n_pose: int, predicate: None=..., max_trial_factor: int = ...) -> TableTopProblemT: ...  # noqa
    # fmt: on

    @classmethod
    def sample(
        cls: Type[TableTopProblemT],
        n_pose: int,
        predicate: Optional[Callable[[TableTopProblemT], bool]] = None,
        max_trial_factor: int = 40,
    ) -> Optional[TableTopProblemT]:
        """
        max_trial_factor is used only if predicate is specified
        """
        pr2 = setup_pr2()
        efkin, colkin = setup_kinmaps()

        def is_collision_init_config(world: TableTopWorld):
            sdf = world.get_union_sdf()
            pts, _ = colkin.map_skrobot_model(pr2)
            vals = sdf(pts[0])
            dists = vals - np.array(colkin.radius_list)
            return np.any(dists < 0.0)

        while True:
            world = TableTopWorld.sample()
            if not is_collision_init_config(world):
                if predicate is None:
                    target_pose_list = [world.sample_pose() for _ in range(n_pose)]
                    problem = cls(world, target_pose_list)
                    return problem
                else:
                    target_pose_list = []
                    trial_count = 0
                    while len(target_pose_list) < n_pose:
                        trial_count += 1
                        pose = world.sample_pose()
                        problem = cls(world, [pose])
                        assert predicate is not None
                        is_valid = predicate(problem)
                        if is_valid:
                            target_pose_list.append(pose)

                        # if sampling the first valid problem takes more than max_trial_factor, then
                        # consider the problem is infeasible
                        seems_infeasible = (
                            len(target_pose_list) == 0 and trial_count > max_trial_factor
                        )
                        if seems_infeasible:
                            return None

                    return cls(world, target_pose_list)

    def visualize(self, av: np.ndarray):
        pr2 = copy.deepcopy(setup_pr2())
        efkin, colkin = setup_kinmaps()
        set_robot_config(pr2, efkin.control_joint_names, av, with_base=True)

        viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
        viewer.add(pr2)
        viewer.add(self.world.table)
        for obs in self.world.obstacles:
            viewer.add(obs)
        for target_pose in self.target_pose_list:
            axis = Axis()
            axis.newcoords(target_pose)
            viewer.add(axis)
        viewer.show()


class _TabletopActualProblem(_TabletopProblem):
    @classmethod
    def get_default_init_solution(cls) -> np.ndarray:
        problem_standard = cls.create_standard()

        n_max_trial = 10
        count = 0
        init_solution: Optional[np.ndarray] = None
        while True:
            try:
                logger.debug("try solving standard problem...")
                result = problem_standard.solve()[0]
                if result.success:
                    logger.debug("solved!")
                    init_solution = result.x
                    break
            except cls.SamplingBasedInitialguessFail:
                pass
            count += 1
            if count > n_max_trial:
                raise RuntimeError("somehow standard problem cannot be solved")
        assert init_solution is not None
        return init_solution


@dataclass
class _TabletopMeshProblem(_TabletopProblem):
    """this problem is actually not a problem.
    This class is for generating data for mesh to train autoencoder.
    So the noth config and result are dummy and will not be used in
    training
    """

    @dataclass
    class DummySolverConfig:
        maxiter: int = -1

    def to_tensors(self) -> torch.Tensor:
        mesh = self.get_mesh()
        mesh_tensor = torch.from_numpy(mesh).float().unsqueeze(dim=0)
        return mesh_tensor

    # fmt: off
    @classmethod
    @overload
    def sample(cls: Type[TableTopProblemT], n_pose: int, predicate: Callable[[TableTopProblemT], bool], max_trial_factor: int = ...) -> Optional[TableTopProblemT]: ...  # noqa

    @classmethod
    @overload
    def sample(cls: Type[TableTopProblemT], n_pose: int, predicate: None, max_trial_factor: int = ...) -> TableTopProblemT: ...  # noqa

    @classmethod
    @overload
    def sample(cls: Type[TableTopProblemT], n_pose: int, predicate: None=..., max_trial_factor: int = ...) -> TableTopProblemT: ...  # noqa
    # fmt: on

    @classmethod
    def sample(
        cls: Type[TableTopProblemT],
        n_pose: int,
        predicate: Optional[Callable[[TableTopProblemT], bool]] = None,
        max_trial_factor: int = 40,
    ) -> Optional[TableTopProblemT]:

        problem: Optional[TableTopProblemT] = super().sample(n_pose, predicate, max_trial_factor)
        if problem is not None:
            problem._aux_gridsdf_cache = problem.grid_sdf
        return problem

    @classmethod
    def get_solver_config(cls) -> DummySolverConfig:
        return cls.DummySolverConfig()

    def solve(self, av_init: Optional[np.ndarray] = None) -> Tuple[()]:
        assert self.n_problem() == 0
        return ()

    @classmethod
    def get_default_init_solution(cls) -> np.ndarray:
        return np.zeros(0)  # whatever


@dataclass
class _TabletopIKProblem(_TabletopActualProblem):
    @classmethod
    def get_solver_config(cls) -> IKConfig:
        config = IKConfig(disp=False)
        return config

    def solve(self, av_init: Optional[np.ndarray] = None) -> Tuple[IKResult, ...]:
        efkin, colkin = setup_kinmaps()
        tspace = TaskSpace(3, sdf=self.get_sdf())  # type: ignore
        cspace = ConfigurationSpace(tspace, colkin, PR2Paramter.rarm_default_bounds(with_base=True))

        result_list = []
        solver_config = self.get_solver_config()
        for target_pose in self.target_pose_list:
            target_pose = skcoords_to_pose_vec(target_pose)
            solver = InverseKinematicsSolver([target_pose], efkin, cspace, config=solver_config)
            result = solver.solve(x_cspace_init=av_init, avoid_obstacle=True)
            result_list.append(result)
        return tuple(result_list)

    def solve_dummy(
        self, av_init: np.ndarray, config: Optional[IKConfig] = None
    ) -> Tuple[ResultProtocol, ...]:
        """solve dummy problem"""

        @dataclass
        class DummyResult:
            success: bool
            nit: int = 0
            nfev: int = 0
            x: np.ndarray = np.zeros(10)

        sdf = self.get_sdf()
        descriptions = self.get_descriptions()
        results = []
        for desc in descriptions:
            point = np.expand_dims(desc[:3], axis=0)
            val = sdf(point)[0]
            results.append(DummyResult(val > 0.0))
        return tuple(results)


@dataclass
class _TabletopPlanningProblem(_TabletopActualProblem):
    @classmethod
    def get_solver_config(cls) -> OsqpSqpPlanner.SolverConfig:
        config = OsqpSqpPlanner.SolverConfig(verbose=False, maxiter=10, maxfev=10)
        return config

    def solve(
        self, traj_vec_init: Optional[np.ndarray] = None
    ) -> Tuple[OsqpSqpPlanner.Result, ...]:
        n_wp = 15
        pr2 = setup_pr2()
        efkin, colkin = setup_kinmaps()
        start = get_robot_config(pr2, efkin.control_joint_names, with_base=True)
        tspace = TaskSpace(3, sdf=self.get_sdf())
        cspace = ConfigurationSpace(tspace, colkin, PR2Paramter.rarm_default_bounds(with_base=True))

        if traj_vec_init is not None:
            traj_init = Trajectory(list(traj_vec_init.reshape(n_wp, -1)))
        else:
            traj_init = None

        solver_config = self.get_solver_config()

        result_list = []
        for target_pose in self.target_pose_list:
            eq_const = TrajectoryEqualityConstraint.from_start(start, n_wp)
            pose_const = PoseConstraint.from_skrobot_coords(target_pose, efkin, cspace=cspace)
            eq_const.add_goal_constraint(pose_const)

            obstacle_const = ObstacleAvoidanceConstraint(cspace)
            ineq_const = TrajectoryInequalityConstraint.create_homogeneous(obstacle_const, n_wp, 10)

            if traj_init is None:
                # creat init traj
                try:
                    samples = batch_sample_from_manifold(
                        20,
                        cspace,
                        eq_const=pose_const,
                        ineq_const=obstacle_const,
                        focus_weight=1.0,
                        max_sample_per_sample=20,
                    )
                except ConstraintSatisfactionFail:
                    raise self.SamplingBasedInitialguessFail

                goal_tree = StartConstraintRRT.from_samples(samples, cspace)
                rrt_config = RRTConfig(n_max_iter=2000)
                rrt = BidirectionalRRT.create_default(start, goal_tree, cspace, rrt_config)
                rrt_solution = rrt.solve()
                if rrt_solution is None:
                    raise self.SamplingBasedInitialguessFail
                traj_init = rrt_solution.resample(n_wp)

            planner = OsqpSqpPlanner(n_wp, eq_const, ineq_const, cspace)
            result = planner.solve(traj_init, solver_config=solver_config)
            result_list.append(result)
        return tuple(result_list)


class ExactGridSDFCreatorMixin:
    def create_gridsdf(self, grid: Grid, sdf: SignedDistanceFunction) -> GridSDF:
        X, Y, Z = grid.get_meshgrid(indexing="ij")
        pts = np.array(list(zip(X.flatten(), Y.flatten(), Z.flatten())))
        values = sdf.__call__(pts)
        gridsdf = GridSDF(grid, values, 2.0, create_itp_lazy=True)
        gridsdf = gridsdf.get_quantized()
        return gridsdf


class VoxbloxGridSDFCreatorMixin:
    def create_gridsdf(self, grid: Grid, sdf: SignedDistanceFunction) -> GridSDF:
        camera = get_pr2_kinect_camera()
        rm_config = RayMarchingConfig(max_dist=2.0)

        esdf = EsdfMap.create(0.02, integrator_type=IntegratorType.MERGED)
        esdf = create_synthetic_esdf(sdf, camera, rm_config=rm_config, esdf=esdf)
        grid_sdf = esdf.get_grid_sdf(grid, fill_value=1.0, create_itp_lazy=True)
        return grid_sdf


# fmt: off
class TabletopMeshProblem(ExactGridSDFCreatorMixin, _TabletopMeshProblem): ...  # noqa
class TabletopIKProblem(ExactGridSDFCreatorMixin, _TabletopIKProblem): ...  # noqa
class TabletopPlanningProblem(ExactGridSDFCreatorMixin, _TabletopPlanningProblem): ...  # noqa
class VoxbloxTabletopMeshProblem(VoxbloxGridSDFCreatorMixin, _TabletopMeshProblem): ...  # noqa
class VoxbloxTabletopIKProblem(VoxbloxGridSDFCreatorMixin, _TabletopIKProblem): ...  # noqa
class VoxbloxTabletopPlanningProblem(VoxbloxGridSDFCreatorMixin, _TabletopPlanningProblem): ...  # noqa
# fmt: on


@dataclass
class CachedMeshFixedProblemPool(FixedProblemPool[TableTopProblemT]):
    dataset: LazyDecomplessDataset
    size: int

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[TableTopProblemT]:
        @dataclass
        class _Iter(Iterator):
            dataset_iter: DatasetIterator
            problem_type: _TabletopMeshProblem
            n_problem_inner: int

            def __next__(self):
                data = next(self.dataset_iter)
                problem: _TabletopProblem = data.cast_to(self.problem_type)
                assert problem.n_problem() == 0
                for _ in range(self.n_problem_inner):
                    pose = problem.world.sample_pose()
                    problem.target_pose_list.append(pose)
                return problem

        return _Iter(self.dataset.__iter__(max_size=self.size), self.problem_type, self.n_problem_inner)  # type: ignore

    @classmethod
    def load(
        cls,
        problem_type: Type[TableTopProblemT],
        mesh_problem_type: Type[_TabletopMeshProblem],
        n_problem_inner: int,
        cache_dir_path: Path,
        size: Optional[int] = None,
    ):
        dataset = LazyDecomplessDataset[_TabletopMeshProblem].load(
            cache_dir_path, mesh_problem_type
        )
        if size is None:
            size = len(dataset)
        assert size <= len(dataset)
        return cls(problem_type, n_problem_inner, dataset, size)
