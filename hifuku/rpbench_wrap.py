import torch
from rpbench.interface import SamplableBase, TaskBase
from rpbench.maze import MazeSolvingTask as _MazeSolvingTask
from rpbench.multiple_rooms import EightRoomsPlanningTask as _EightRoomsPlanningTask
from rpbench.pr2.kivapod import KivapodEmptyReachingTask as _KivapodEmptyReachingTask

# fmt: off
from rpbench.pr2.tabletop import (
    TabletopBoxDualArmReachingTask as _TabletopBoxDualArmReachingTask,
)
from rpbench.pr2.tabletop import (
    TabletopBoxRightArmReachingTask as _TabletopBoxRightArmReachingTask,
)
from rpbench.pr2.tabletop import (
    TabletopBoxVoxbloxRightArmReachingTask as _TabletopBoxVoxbloxRightArmReachingTask,
)
from rpbench.pr2.tabletop import TabletopBoxWorldWrap as _TabletopBoxWorldWrap
from rpbench.pr2.tabletop import (
    TabletopVoxbloxBoxWorldWrap as _TabletopVoxbloxBoxWorldWrap,
)
from rpbench.ring import (
    RingObstacleFreeBlockedPlanningTask as _RingObstacleFreeBlockedPlanningTask,
)
from rpbench.ring import RingObstacleFreePlanningTask as _RingObstacleFreePlanningTask

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
class KivapodEmptyReachingTask(_KivapodEmptyReachingTask, PicklableTaskBase): ...  # noqa
class TabletopBoxRightArmReachingTask(_TabletopBoxRightArmReachingTask, PicklableTaskBase): ...  # noqa
class TabletopBoxDualArmReachingTask(_TabletopBoxDualArmReachingTask, PicklableTaskBase): ...  # noqa
class TabletopBoxVoxbloxRightArmReachingTask(_TabletopBoxVoxbloxRightArmReachingTask, PicklableTaskBase): ...  # noqa
class TabletopBoxWorldWrap(_TabletopBoxWorldWrap, PicklableTensorExportSamplableBase): ...  # noqa
class TabletopVoxbloxBoxWorldWrap(_TabletopVoxbloxBoxWorldWrap, PicklableTensorExportSamplableBase): ...  # noqa
class MazeSolvingTask(_MazeSolvingTask, PicklableTaskBase): ...  # noqa
class RingObstacleFreePlanningTask(_RingObstacleFreePlanningTask, PicklableTaskBase): ...  # noqa
class RingObstacleFreeBlockedPlanningTask(_RingObstacleFreeBlockedPlanningTask, PicklableTaskBase): ...  # noqa
class EightRoomsPlanningTask(_EightRoomsPlanningTask, PicklableTaskBase): ...  # noqa
# fmt: on
