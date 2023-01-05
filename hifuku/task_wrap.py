import torch
from rpbench.interface import SamplableBase, TaskBase
from rpbench.tabletop import (
    TabletopBoxRightArmReachingTask as _TabletopBoxRightArmReachingTask,
)
from rpbench.tabletop import TabletopBoxWorldWrap as _TabletopBoxWorldWrap

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


class TabletopBoxRightArmReachingTask(_TabletopBoxRightArmReachingTask, PicklableTaskBase):
    ...


class TabletopBoxWorldWrap(_TabletopBoxWorldWrap, PicklableTensorExportSamplableBase):
    ...
