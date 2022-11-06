from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from skrobot.model.link import Link
from skrobot.model.primitives import Box, Cylinder
from skrobot.sdf import UnionSDF
from voxbloxpy.core import Grid, GridSDF


@dataclass
class TableTopWorld:
    table: Box
    obstacles: List[Link]

    def get_union_sdf(self) -> UnionSDF:
        plane = Box([30, 30, 0.1], pos=(0, 0, -0.05), with_sdf=True)
        lst = [self.table.sdf, plane.sdf]
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

        n_box = np.random.randint(3)
        n_cylinder = np.random.randint(8)

        obstacles = []

        for _ in range(n_box):
            dimension = np.array([0.1, 0.1, 0.05]) + np.random.rand(3) * np.array([0.2, 0.2, 0.2])
            box = Box(extents=dimension, with_sdf=True)

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
            cylinder = Cylinder(radius=r, height=h, with_sdf=True)

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


class TabletopIKProblem:
    world: TableTopWorld
    grid_sdf: GridSDF
