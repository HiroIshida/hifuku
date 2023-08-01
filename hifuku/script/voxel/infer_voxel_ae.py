import numpy as np
import torch
from mohou.file import get_project_path
from mohou.trainer import TrainCache
from rpbench.articulated.pr2.tabletop import TabletopBoxWorldWrap
from voxbloxpy.core import Grid, GridSDF

from hifuku.neuralnet import VoxelAutoEncoder


def render(mesh):
    grid = Grid(lb=np.zeros(3), ub=np.ones(3), sizes=mesh.shape)
    sdf = GridSDF(grid, mesh.flatten(), 2.0)
    sdf.render_volume()


if __name__ == "__main__":
    samplable_type = TabletopBoxWorldWrap
    pp = get_project_path("hifuku-{}".format(samplable_type.__name__))
    best_model = TrainCache.load(pp, VoxelAutoEncoder).best_model

    problem = samplable_type.sample(0, standard=False)
    gridsdf = problem.cache
    assert gridsdf is not None
    mesh = gridsdf.values.reshape(*gridsdf.grid.sizes)

    sample = torch.from_numpy(np.expand_dims(np.expand_dims(mesh, axis=0), axis=0)).float()
    out = best_model.decoder(best_model.encoder(sample))
    mesh_reconstruct = out.detach().numpy()[0][0]

    render(mesh)
    render(mesh_reconstruct)
