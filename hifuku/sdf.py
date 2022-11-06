from typing import Callable, Tuple

import numpy as np
from skplan.sdf import SDFLike

SDFProtocol = SDFLike


def create_union_sdf(sdfs: Tuple[SDFProtocol, ...]) -> Callable[[np.ndarray], np.ndarray]:
    def f(pts: np.ndarray):
        values = np.array([sdf(pts) for sdf in sdfs])
        values_min = np.min(values, axis=0)
        return values_min

    return f
