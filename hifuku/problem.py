from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from skrobot.model.primitives import Box


@dataclass
class CollisionAvoidanceIKProblem:
    target_pose: np.ndarray  # 6 dim
    sdf_mesh: np.ndarray
    grid_size: Tuple[int, int, int]
    _original_obstacles: Optional[List[Box]] = None

    @classmethod
    def create(
        cls, pose: np.ndarray, primary_obstacle: Box, obstacles: List[Box]
    ) -> "CollisionAvoidanceIKProblem":
        pass
