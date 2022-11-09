import time

import numpy as np
import skrobot
from skplan.robot.pr2 import PR2Paramter
from skplan.solver.optimization import IKConfig, InverseKinematicsSolver
from skplan.space import ConfigurationSpace, TaskSpace
from skplan.viewer.skrobot_viewer import (
    CollisionSphereVisualizationManager,
    set_robot_config,
)
from skrobot.model.primitives import Axis, PointCloudLink
from voxbloxpy.core import EsdfMap, IntegratorType

from hifuku.sdf import create_union_sdf
from hifuku.threedim.camera import RayMarchingConfig, create_synthetic_esdf
from hifuku.threedim.robot import get_pr2_kinect_camera
from hifuku.threedim.tabletop import create_simple_tabletop_world
from hifuku.threedim.utils import skcoords_to_pose_vec

np.random.seed(4)
use_base = True

robot_model = skrobot.models.PR2()
robot_model.reset_manip_pose()

# create grid sdf using voxbloxpy
world = create_simple_tabletop_world(with_obstacle=True)
union_sdf = world.get_union_sdf()
camera = get_pr2_kinect_camera()
esdf = EsdfMap.create(0.02, integrator_type=IntegratorType.MERGED)
esdf = create_synthetic_esdf(
    union_sdf, camera, rm_config=RayMarchingConfig(max_dist=2.0), esdf=esdf
)
grid_sdf = esdf.get_grid_sdf(world.get_grid(), fill_value=1.0, create_itp_lazy=True)
grid_sdf_comp = grid_sdf.get_quantized()

sdf = create_union_sdf((grid_sdf_comp, world.table.sdf))  # type: ignore

col_kinmap = PR2Paramter.collision_kinematics(with_base=use_base)
col_kinmap.reflect_skrobot_model(robot_model)
tspace = TaskSpace(3, sdf=sdf)  # type: ignore
cspace = ConfigurationSpace(tspace, col_kinmap, PR2Paramter.rarm_default_bounds(with_base=use_base))
kinmap = PR2Paramter.rarm_kinematics(with_base=use_base)
kinmap.reflect_skrobot_model(robot_model)

target_co = Axis(axis_radius=0.001, axis_length=0.05)
target_co.translate([0.85, -0.05, 0.9])
target_co.rotate(0.4, axis="z")
target_pose = skcoords_to_pose_vec(target_co)

ik_config = IKConfig(clearance=0.05)
solver = InverseKinematicsSolver([target_pose], kinmap, cspace, config=ik_config)

res = solver.solve(avoid_obstacle=True)

visualize = True
if visualize:
    set_robot_config(robot_model, kinmap.control_joint_names, res.x, with_base=use_base)

    voxel_info = esdf.get_voxel_info().filter(0.0, 0.02)
    voxel_origin_obstacles = voxel_info.origins
    plink = PointCloudLink(voxel_origin_obstacles)

    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
    viewer.add(world.table)
    viewer.add(robot_model)
    viewer.add(camera)
    viewer.add(target_co)
    viewer.add(plink)
    for obs in world.obstacles:
        viewer.add(obs)

    colvis_manager = CollisionSphereVisualizationManager(col_kinmap, viewer)
    colvis_manager.update(robot_model, sdf)  # type: ignore
    viewer.show()
    # grid_sdf = esdf.get_grid_sdf(esdf.get_voxel_info().get_boundary_grid(0.02), fill_value=1.0)
    # grid_sdf.render_volume(isomin=-0.1, isomax=0.1)
    time.sleep(30)
