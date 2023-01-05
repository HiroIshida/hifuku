from rpbench.interface import TaskBase
from rpbench.tabletop import (
    TabletopBoxRightArmReachingTask as _TabletopBoxRightArmReachingTask,
)

from hifuku.llazy.dataset import PicklableChunkBase


class PicklableTaskBase(TaskBase, PicklableChunkBase):
    ...


class TabletopBoxRightArmReachingTask(_TabletopBoxRightArmReachingTask, PicklableTaskBase):
    ...
