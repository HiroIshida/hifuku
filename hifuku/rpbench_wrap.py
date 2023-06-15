import torch
from rpbench.interface import SamplableBase, TaskBase
from rpbench.jaxon.below_table import (
    HumanoidTableReachingTask as _HumanoidTableReachingTask,
)
from rpbench.multiple_rooms import EightRoomsPlanningTask as _EightRoomsPlanningTask
from rpbench.pr2.kivapod import KivapodEmptyReachingTask as _KivapodEmptyReachingTask

# fmt: off
from rpbench.pr2.tabletop import (
    TabletopBoxDualArmReachingTask as _TabletopBoxDualArmReachingTask,
)
from rpbench.pr2.tabletop import (
    TabletopOvenDualArmReachingTask as _TabletopOvenDualArmReachingTask,
)
from rpbench.pr2.tabletop import (
    TabletopOvenRightArmReachingTask as _TabletopOvenRightArmReachingTask,
)
from rpbench.pr2.tabletop import (
    TabletopOvenVoxbloxRightArmReachingTask as _TabletopOvenVoxbloxRightArmReachingTask,
)
from rpbench.pr2.tabletop import (
    TabletopOvenVoxbloxWorldWrap as _TabletopOvenVoxbloxWorldWrap,
)
from rpbench.pr2.tabletop import TabletopOvenWorldWrap as _TabletopOvenWorldWrap
from rpbench.ring import (
    RingObstacleFreeBlockedPlanningTask as _RingObstacleFreeBlockedPlanningTask,
)
from rpbench.ring import RingObstacleFreePlanningTask as _RingObstacleFreePlanningTask
from rpbench.two_dimensional.bubbly_world import (
    BubblyComplexMeshPointConnectTask as _BubblyComplexMeshPointConnectTask,
)
from rpbench.two_dimensional.bubbly_world import (
    BubblyComplexPointConnectTask as _BubblyComplexPointConnectTask,
)
from rpbench.two_dimensional.bubbly_world import (
    BubblySimpleMeshPointConnectTask as _BubblySimpleMeshPointConnectTask,
)
from rpbench.two_dimensional.bubbly_world import (
    BubblySimplePointConnectTask as _BubblySimplePointConnectTask,
)
from rpbench.two_dimensional.maze import MazeSolvingTask as _MazeSolvingTask

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
class TabletopOvenRightArmReachingTask(_TabletopOvenRightArmReachingTask, PicklableTaskBase): ...  # noqa
class TabletopOvenDualArmReachingTask(_TabletopOvenDualArmReachingTask, PicklableTaskBase): ...  # noqa
class TabletopOvenVoxbloxRightArmReachingTask(_TabletopOvenVoxbloxRightArmReachingTask, PicklableTaskBase): ...  # noqa
class TabletopOvenWorldWrap(_TabletopOvenWorldWrap, PicklableTensorExportSamplableBase): ...  # noqa
class TabletopOvenVoxbloxWorldWrap(_TabletopOvenVoxbloxWorldWrap, PicklableTensorExportSamplableBase): ...  # noqa
class TabletopBoxDualArmReachingTask(_TabletopBoxDualArmReachingTask, PicklableTaskBase): ...  # noqa
class MazeSolvingTask(_MazeSolvingTask, PicklableTaskBase): ...  # noqa
class RingObstacleFreePlanningTask(_RingObstacleFreePlanningTask, PicklableTaskBase): ...  # noqa
class RingObstacleFreeBlockedPlanningTask(_RingObstacleFreeBlockedPlanningTask, PicklableTaskBase): ...  # noqa
class EightRoomsPlanningTask(_EightRoomsPlanningTask, PicklableTaskBase): ...  # noqa
class HumanoidTableReachingTask(_HumanoidTableReachingTask, PicklableTaskBase): ...  # noqa
class BubblySimpleMeshPointConnectTask(_BubblySimpleMeshPointConnectTask, PicklableTaskBase): ...  # noqa
class BubblyComplexMeshPointConnectTask(_BubblyComplexMeshPointConnectTask, PicklableTaskBase): ...  # noqa
class BubblySimplePointConnectTask(_BubblySimplePointConnectTask, PicklableTaskBase): ...  # noqa
class BubblyComplexPointConnectTask(_BubblyComplexPointConnectTask, PicklableTaskBase): ...  # noqa
# fmt: on
