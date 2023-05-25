import time

import numpy as np
import torch
import trimesh
from mohou.file import get_project_path
from skimage import measure
from skrobot.model import Box
from skrobot.model.primitives import Axis, LineString, MeshLink
from skrobot.viewers import TrimeshSceneViewer

from hifuku.domain import (
    HumanoidTableRarmReaching_SQP_Domain,
    HumanoidTableReachingTask,
)
from hifuku.library import SolutionLibrary

if __name__ == "__main__":
    domain = HumanoidTableRarmReaching_SQP_Domain
    task_type = domain.task_type
    solver_type = domain.solver_type
    interactive = False

    pp = get_project_path("tabletop_solution_library-{}".format(domain.get_domain_name()))
    libraries = SolutionLibrary.load(pp, task_type, solver_type, torch.device("cpu"))
    lib = libraries[0]
    task = lib.task_type.sample(1, standard=True)
    region: Box = task.world.target_region

    n_grid = 50
    region_center = region.worldpos()
    extent = np.array(region._extents)
    margin = 0.0
    lb = region_center - (0.5 + margin) * extent
    ub = region_center + (0.5 + margin) * extent
    xlin, ylin, zlin = [np.linspace(lb[i], ub[i], n_grid) for i in range(3)]
    X, Y, Z = np.meshgrid(xlin, ylin, zlin)
    pts = np.array(list(zip(X.flatten(), Y.flatten(), Z.flatten())))

    table = task.export_table()

    dummy_desc = torch.empty((len(pts), 0))
    vec = torch.from_numpy(table.get_vector_descs()[0]).float().unsqueeze(0)
    vecs = vec.repeat(len(pts), 1)
    vecs[:, 6:9] = torch.from_numpy(pts).float()

    task: HumanoidTableReachingTask
    config = task.config_provider.get_config()
    efkin = config.get_endeffector_kin()
    jaxon = task.config_provider.get_jaxon()

    mesh_links = []
    line_links = []
    idx = 0
    for idx in range(len(lib.predictors)):
        pred = lib.predictors[idx]
        values = lib.success_iter_threshold() - (
            pred.forward((dummy_desc, vecs))[0] + lib.margins[idx]
        )
        values = values.detach().numpy()
        if np.max(values) > 0.0:
            spacing = (ub - lb) / (n_grid - 1)
            F = values.reshape(n_grid, n_grid, n_grid)
            F = np.swapaxes(F, 0, 1)  # important!!!
            verts, faces, _, _ = measure.marching_cubes_lewiner(F, 0, spacing=spacing)
            verts = verts + lb
            faces = faces[:, ::-1]

            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            mesh = trimesh.smoothing.filter_laplacian(mesh)

            mesh_link = MeshLink(mesh)
            rgb = np.random.randint(0, 255, 3)
            mesh_link.visual_mesh.visual.face_colors[:, 0] = rgb[0]
            mesh_link.visual_mesh.visual.face_colors[:, 1] = rgb[1]
            mesh_link.visual_mesh.visual.face_colors[:, 2] = rgb[2]
            mesh_link.visual_mesh.visual.face_colors[:, 3] = 150
            mesh_links.append(mesh_link)

            traj = pred.initial_solution

            feature_pointss, _ = efkin.map(traj.numpy())

            n_wp, n_feature, n_tspace = feature_pointss.shape
            feature_points = feature_pointss[:, 2, :]

            pts = []
            for feature_point in feature_points:
                pt = feature_point[:3]
                pts.append(pt)
            print(pts)

            line_link = LineString(np.array(pts))
            line_link.visual_mesh.colors = [[rgb[0], rgb[1], rgb[2], 255]]
            line_links.append(line_link)

    vis = TrimeshSceneViewer()
    task.world.visualize(vis)
    co = task.descriptions[0][0]
    axis = Axis.from_coords(co)
    vis.add(jaxon)
    vis.add(axis)
    for link in line_links:
        vis.add(link)

    for mesh_link in mesh_links:
        vis.add(mesh_link)
    vis.redraw()
    vis.show()
    time.sleep(100)
