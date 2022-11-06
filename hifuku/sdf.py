from typing import Protocol, Tuple

import numpy as np


class SDFProtocol(Protocol):
    def __call__(self, pts: np.ndarray) -> np.ndarray:
        ...


def create_union_sdf(sdfs: Tuple[SDFProtocol, ...]) -> SDFProtocol:
    def f(pts: np.ndarray):
        values = np.array([sdf(pts) for sdf in sdfs])
        values_min = np.min(values, axis=0)
        return values_min

    return f
