import argparse
from enum import Enum
from typing import Type

import numpy as np
import torch
from mohou.file import get_project_path
from mohou.trainer import TrainCache
from voxbloxpy.core import Grid, GridSDF

from hifuku.neuralnet import VoxelAutoEncoder
from hifuku.threedim.tabletop import (
    TabletopMeshProblem,
    TableTopWorld,
    VoxbloxTabletopMeshProblem,
    _TabletopProblem,
)


class ProblemType(Enum):
    normal = TabletopMeshProblem
    voxblox = VoxbloxTabletopMeshProblem


def render(mesh):
    grid = Grid(lb=np.zeros(3), ub=np.ones(3), sizes=mesh.shape)
    sdf = GridSDF(grid, mesh.flatten(), 2.0)
    sdf.render_volume()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", type=str, default="normal", help="")
    args = parser.parse_args()
    mesh_type_name: str = args.type

    problem_type: Type[_TabletopProblem] = ProblemType[mesh_type_name].value
    pp = get_project_path("tabletop_mesh-{}".format(problem_type.__name__))
    best_model = TrainCache.load(pp, VoxelAutoEncoder).best_model

    problem = VoxbloxTabletopMeshProblem.sample(0)
    world = TableTopWorld.sample()
    gridsdf = problem.grid_sdf
    mesh = gridsdf.values.reshape(*gridsdf.grid.sizes)

    sample = torch.from_numpy(np.expand_dims(np.expand_dims(mesh, axis=0), axis=0)).float()
    out = best_model.decoder(best_model.encoder(sample))
    mesh_reconstruct = out.detach().numpy()[0][0]

    render(mesh)
    render(mesh_reconstruct)
