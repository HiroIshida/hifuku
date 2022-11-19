import numpy as np
import torch
from mohou.file import get_project_path
from mohou.trainer import TrainCache
from voxbloxpy.core import Grid, GridSDF

from hifuku.neuralnet import VoxelAutoEncoder
from hifuku.threedim.tabletop import TableTopWorld


def render(mesh):
    grid = Grid(lb=np.zeros(3), ub=np.ones(3), sizes=mesh.shape)
    sdf = GridSDF(grid, mesh.flatten(), 2.0)
    sdf.render_volume()


pp = get_project_path("tabletop_mesh")
best_model = TrainCache.load(pp, VoxelAutoEncoder).best_model

world = TableTopWorld.sample()
gridsdf = world.compute_exact_gridsdf(fill_value=2.0)
gridsdf = gridsdf.get_quantized()
mesh = gridsdf.values.reshape(*gridsdf.grid.sizes)

sample = torch.from_numpy(np.expand_dims(np.expand_dims(mesh, axis=0), axis=0)).float()
out = best_model.decoder(best_model.encoder(sample))
mesh_reconstruct = out.detach().numpy()[0][0]

render(mesh)
render(mesh_reconstruct)
