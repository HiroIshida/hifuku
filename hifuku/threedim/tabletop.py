import copy
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Type, TypeVar

import numpy as np
import skrobot
from skplan.kinematics import (
    ArticulatedCollisionKinematicsMap,
    ArticulatedEndEffectorKinematicsMap,
)
from skplan.robot.pr2 import PR2Paramter
from skplan.solver.constraint import PoseConstraint, TrajectoryEqualityConstraint
from skplan.solver.inverse_kinematics import IKConfig, IKResult, InverseKinematicsSolver
from skplan.solver.optimization import (
    OptimizationBasedPlanner,
    PlannerConfig,
    PlanningResult,
)
from skplan.space import ConfigurationSpace, TaskSpace
from skplan.trajectory import Trajectory
from skplan.viewer.skrobot_viewer import get_robot_config, set_robot_config
from skrobot.coordinates import Coordinates
from skrobot.model import Axis
from skrobot.model.link import Link
from skrobot.model.primitives import Box, Cylinder
from skrobot.models.pr2 import PR2
from skrobot.sdf import UnionSDF
from voxbloxpy.core import Grid, GridSDF

from hifuku.sdf import create_union_sdf
from hifuku.threedim.utils import skcoords_to_pose_vec
from hifuku.types import ProblemInterface, ResultProtocol


@dataclass
class TableTopWorld:
    table: Box
    obstacles: List[Link]

    def get_union_sdf(self) -> UnionSDF:
        lst = [self.table.sdf]
        for obstacle in self.obstacles:
            lst.append(obstacle.sdf)
        return UnionSDF(lst)

    def get_grid(
        self, grid_sizes: Tuple[int, int, int] = (56, 56, 28), mesh_height: float = 0.3
    ) -> Grid:
        depth, width, height = self.table._extents
        lb = np.array([-0.5 * depth, -0.5 * width, 0.5 * height - 0.1])
        ub = np.array([+0.5 * depth, +0.5 * width, 0.5 * height + mesh_height])
        lb = self.table.transform_vector(lb)
        ub = self.table.transform_vector(ub)
        return Grid(lb, ub, grid_sizes)

    def compute_exact_gridsdf(
        self,
        grid_sizes: Tuple[int, int, int] = (56, 56, 28),
        mesh_height: float = 0.3,
        fill_value: float = np.nan,
    ) -> GridSDF:

        grid = self.get_grid()
        X, Y, Z = grid.get_meshgrid(indexing="ij")
        pts = np.array(list(zip(X.flatten(), Y.flatten(), Z.flatten())))

        sdf = UnionSDF([obs.sdf for obs in self.obstacles])
        values = sdf.__call__(pts)
        return GridSDF(grid, values, fill_value, create_itp_lazy=True)

    def sample_pose(self) -> Coordinates:
        table = self.table
        table_depth, table_width, table_height = table._extents
        table_tip = table.copy_worldcoords()
        table_tip.translate([-table_depth * 0.5, -table_width * 0.5, +0.5 * table_height])
        table_tip.translate([0, 0, 0.05])

        diff = np.random.rand(3) * np.array([table_depth, table_width, 0.2])
        table_tip.translate(diff)
        table_tip.rotate(-1.0 + np.random.rand() * 2.0, axis="z")
        return table_tip.copy_worldcoords()

    @classmethod
    def sample(cls) -> "TableTopWorld":
        table = cls.create_standard_table()
        table_depth, table_width, table_height = table._extents
        x = np.random.rand() * 0.2
        y = -0.2 + np.random.rand() * 0.4
        z = 0.0
        table.translate([x, y, z])

        table_tip = table.copy_worldcoords()
        table_tip.translate([-table_depth * 0.5, -table_width * 0.5, +0.5 * table_height])

        n_box = np.random.randint(3) + 1
        n_cylinder = np.random.randint(8) + 1

        obstacles = []

        color = np.array([255, 0, 0, 200])

        for _ in range(n_box):
            dimension = np.array([0.1, 0.1, 0.05]) + np.random.rand(3) * np.array([0.2, 0.2, 0.2])
            box = Box(extents=dimension, with_sdf=True, face_colors=color)

            co = table_tip.copy_worldcoords()
            box.newcoords(co)
            x = dimension[0] * 0.5 + np.random.rand() * (table_depth - dimension[0])
            y = dimension[1] * 0.5 + np.random.rand() * (table_width - dimension[1])
            z = dimension[2] * 0.5
            box.translate([x, y, z])
            obstacles.append(box)

        for _ in range(n_cylinder):
            r = np.random.rand() * 0.03 + 0.01
            h = np.random.rand() * 0.2 + 0.05
            cylinder = Cylinder(radius=r, height=h, with_sdf=True, face_colors=color)

            co = table_tip.copy_worldcoords()
            cylinder.newcoords(co)
            x = r + np.random.rand() * (table_depth - r)
            y = r + np.random.rand() * (table_width - r)
            z = 0.5 * h
            cylinder.translate([x, y, z])
            obstacles.append(cylinder)

        return cls(table, obstacles)

    @staticmethod
    def create_standard_table() -> Box:
        # create jsk-lab 73b2 table
        table_depth = 0.5
        table_width = 0.75
        table_height = 0.7
        pos = [0.5 + table_depth * 0.5, 0.0, table_height * 0.5]
        table = Box(extents=[table_depth, table_width, table_height], pos=pos, with_sdf=True)
        return table


def create_simple_tabletop_world(with_obstacle: bool = False) -> TableTopWorld:
    table = TableTopWorld.create_standard_table()
    table_depth, table_width, table_height = table._extents

    table_tip = table.copy_worldcoords()
    table_tip.translate([-table_depth * 0.5, -table_width * 0.5, +0.5 * table_height])

    obstacles = []
    if with_obstacle:
        box_co = table_tip.copy_worldcoords()
        box_co.translate([0.2, 0.5, 0.1])
        box = Box(extents=[0.1, 0.2, 0.2], with_sdf=True)
        box.newcoords(box_co)
        obstacles.append(box)

        cylinder_co = table_tip.copy_worldcoords()
        cylinder_co.translate([0.45, 0.1, 0.15])
        cylinder = Cylinder(radius=0.05, height=0.3, with_sdf=True)
        cylinder.newcoords(cylinder_co)
        obstacles.append(cylinder)

        cylinder_co = table_tip.copy_worldcoords()
        cylinder_co.translate([0.0, 0.1, 0.15])
        cylinder = Cylinder(radius=0.05, height=0.3, with_sdf=True)
        cylinder.newcoords(cylinder_co)
        obstacles.append(cylinder)

    return TableTopWorld(table, obstacles)


_cache = {"kinmap": None, "pr2": None}


TableTopProblemT = TypeVar("TableTopProblemT", bound="TabletopProblem")


@dataclass
class TabletopProblem(ProblemInterface):
    world: TableTopWorld
    grid_sdf: GridSDF
    target_pose_list: List[Coordinates]

    @classmethod
    def setup_pr2(cls) -> PR2:
        if _cache["pr2"] is None:
            pr2 = PR2(use_tight_joint_limit=False)
            pr2.reset_manip_pose()
            _cache["pr2"] = pr2
        return _cache["pr2"]  # type: ignore

    @classmethod
    def setup_kinmaps(
        cls,
    ) -> Tuple[ArticulatedEndEffectorKinematicsMap, ArticulatedCollisionKinematicsMap]:
        if _cache["kinmap"] is None:
            pr2 = cls.setup_pr2()
            efkin = PR2Paramter.rarm_kinematics(with_base=True)
            efkin.reflect_skrobot_model(pr2)
            colkin = PR2Paramter.collision_kinematics(with_base=True)
            colkin.reflect_skrobot_model(pr2)
            _cache["kinmap"] = (efkin, colkin)  # type: ignore
        return _cache["kinmap"]  # type: ignore

    def get_sdf(self) -> Callable[[np.ndarray], np.ndarray]:
        sdf = create_union_sdf((self.grid_sdf, self.world.table.sdf))  # type: ignore
        return sdf

    def get_descriptions(self) -> List[np.ndarray]:
        table_pose = skcoords_to_pose_vec(self.world.table.worldcoords())
        # description vector is composed of 6 + 6 dimension
        return [
            np.hstack([skcoords_to_pose_vec(pose), table_pose]) for pose in self.target_pose_list
        ]

    def get_mesh(self) -> np.ndarray:
        grid_sdf = self.grid_sdf
        return grid_sdf.values.reshape(grid_sdf.grid.sizes)

    @classmethod
    def get_description_dim(cls) -> int:
        return 12

    @classmethod
    def sample(cls: Type[TableTopProblemT], n_pose: int) -> TableTopProblemT:
        pr2 = cls.setup_pr2()
        efkin, colkin = cls.setup_kinmaps()

        def is_collision_init_config(world: TableTopWorld):
            sdf = world.get_union_sdf()
            pts, _ = colkin.map_skrobot_model(pr2)
            vals = sdf(pts[0])
            dists = vals - np.array(colkin.radius_list)
            return np.any(dists < 0.0)

        while True:
            world = TableTopWorld.sample()
            if not is_collision_init_config(world):
                gridsdf = world.compute_exact_gridsdf(fill_value=2.0)
                gridsdf = gridsdf.get_quantized()
                target_pose_list = [world.sample_pose() for _ in range(n_pose)]
                problem = cls(world, gridsdf, target_pose_list)
                return problem

    def visualize(self, av: np.ndarray):
        pr2 = copy.deepcopy(self.setup_pr2())
        efkin, colkin = self.setup_kinmaps()
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


@dataclass
class TabletopIKProblem(TabletopProblem):
    def solve(
        self, av_init: Optional[np.ndarray] = None, config: Optional[IKConfig] = None
    ) -> Tuple[IKResult, ...]:
        if config is None:
            config = IKConfig()
        efkin, colkin = self.setup_kinmaps()
        tspace = TaskSpace(3, sdf=self.get_sdf())  # type: ignore
        cspace = ConfigurationSpace(tspace, colkin, PR2Paramter.rarm_default_bounds(with_base=True))

        result_list = []
        for target_pose in self.target_pose_list:
            target_pose = skcoords_to_pose_vec(target_pose)
            solver = InverseKinematicsSolver([target_pose], efkin, cspace, config=config)
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
class TabletopPlanningProblem(TabletopProblem):
    def solve(
        self, traj_vec_init: Optional[np.ndarray] = None, config: Optional[PlannerConfig] = None
    ) -> Tuple[PlanningResult, ...]:

        if config is None:
            config = PlannerConfig()

        n_wp = 10
        pr2 = self.setup_pr2()
        efkin, colkin = self.setup_kinmaps()
        start = get_robot_config(pr2, efkin.control_joint_names, with_base=True)

        tspace = TaskSpace(3, sdf=self.get_sdf())  # type: ignore
        cspace = ConfigurationSpace(tspace, colkin, PR2Paramter.rarm_default_bounds(with_base=True))

        if traj_vec_init is not None:
            traj_init = Trajectory(list(traj_vec_init.reshape(n_wp, -1)))
        else:
            traj_init = None

        result_list = []
        for target_pose in self.target_pose_list:
            eq_const = TrajectoryEqualityConstraint.from_start(start, 10)
            pose_const = PoseConstraint.from_skrobot_coords(target_pose, efkin, cspace=cspace)
            eq_const.add_goal_constraint(pose_const)
            planner = OptimizationBasedPlanner(eq_const, cspace, config=config)
            result = planner.solve(traj_init)
            result_list.append(result)
        return tuple(result_list)
