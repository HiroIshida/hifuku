import torch
from rpbench.interface import SamplableBase, TaskBase
from rpbench.maze import MazeSolvingTask as _MazeSolvingTask

# fmt: off
from rpbench.tabletop import (
    TabletopBoxDualArmReachingTask as _TabletopBoxDualArmReachingTask,
)
from rpbench.tabletop import (
    TabletopBoxRightArmReachingTask as _TabletopBoxRightArmReachingTask,
)
from rpbench.tabletop import (
    TabletopBoxVoxbloxRightArmReachingTask as _TabletopBoxVoxbloxRightArmReachingTask,
)
from rpbench.tabletop import TabletopBoxWorldWrap as _TabletopBoxWorldWrap
from rpbench.tabletop import TabletopVoxbloxBoxWorldWrap as _TabletopVoxbloxBoxWorldWrap

# fmt: on
from hifuku.llazy.dataset import PicklableChunkBase, PicklableTensorChunkBase


class PicklableSamplableBase(SamplableBase, PicklableChunkBase):
    """SamplableBase with picklability
    Main purpose of this class is to use this in llazy's dataset
    """

    ...


class PicklableTaskBase(TaskBase, PicklableSamplableBase):
    """TaskBase with picklability
    Main purpose of this class is to use this in llazy's dataset
    """

    ...


class PicklableTensorExportSamplableBase(PicklableTensorChunkBase, PicklableSamplableBase):
    """TaskBase with picklability and tensor exportability
    Main purpose of this class is to use this in llazy's dataloader and
    export the world mesh as tensor
    """

    def to_tensors(self) -> torch.Tensor:
        description_table = self.export_table()
        mesh = description_table.get_mesh()
        mesh_tensor = torch.from_numpy(mesh).float().unsqueeze(dim=0)
        return mesh_tensor


# fmt: off
class TabletopBoxRightArmReachingTask(_TabletopBoxRightArmReachingTask, PicklableTaskBase): ...  # noqa
class TabletopBoxDualArmReachingTask(_TabletopBoxDualArmReachingTask, PicklableTaskBase): ...  # noqa
class TabletopBoxVoxbloxRightArmReachingTask(_TabletopBoxVoxbloxRightArmReachingTask, PicklableTaskBase): ...  # noqa
class TabletopBoxWorldWrap(_TabletopBoxWorldWrap, PicklableTensorExportSamplableBase): ...  # noqa
class TabletopVoxbloxBoxWorldWrap(_TabletopVoxbloxBoxWorldWrap, PicklableTensorExportSamplableBase): ...  # noqa
class MazeSolvingTask(_MazeSolvingTask, PicklableTaskBase): ...  # noqa
# fmt: on
