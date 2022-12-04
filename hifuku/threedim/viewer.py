import collections
from typing import Dict

import trimesh
from skrobot.model import CascadedLink, Link, RobotModel


class SceneWrapper(trimesh.Scene):
    _links: Dict[str, Link]

    def __init__(self):
        super(SceneWrapper, self).__init__()
        self._links = collections.OrderedDict()

    def show(self):
        pass

    def redraw(self):
        # apply latest angle-vector
        for link_id, link in self._links.items():
            link.update(force=True)
            transform = link.worldcoords().T()
            self.graph.update(link_id, matrix=transform)

    def update_scene_graph(self):
        # apply latest angle-vector
        for link_id, link in self._links.items():
            link.update(force=True)
            transform = link.worldcoords().T()
            self.graph.update(link_id, matrix=transform)

    @staticmethod
    def convert_geometry_to_links(geometry):
        if isinstance(geometry, Link):
            links = [geometry]
        elif isinstance(geometry, CascadedLink):
            links = geometry.link_list
        else:
            raise TypeError("geometry must be Link or CascadedLink")
        return links

    def add(self, link):
        links = self.convert_geometry_to_links(link)

        for link in links:
            link_id = str(id(link))
            if link_id in self._links:
                return
            transform = link.worldcoords().T()
            mesh = link.visual_mesh
            # TODO(someone) fix this at trimesh's scene.
            if (isinstance(mesh, list) or isinstance(mesh, tuple)) and len(mesh) > 0:
                mesh = trimesh.util.concatenate(mesh)
            self.add_geometry(
                geometry=mesh,
                node_name=link_id,
                geom_name=link_id,
                transform=transform,
            )
            self._links[link_id] = link


def set_robot_alpha(robot: RobotModel, alpha: int):
    assert alpha < 256
    for link in robot.link_list:
        visual_mesh = link.visual_mesh
        if isinstance(visual_mesh, list):
            for mesh in visual_mesh:
                mesh.visual.face_colors[:, 3] = alpha
        else:
            visual_mesh.visual.face_colors[:, 3] = alpha
