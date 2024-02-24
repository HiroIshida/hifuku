# this file implements some class for testing

from dataclasses import dataclass

from rpbench.articulated.pr2.minifridge import TabletopClutteredFridgeReachingTask


@dataclass
class SimplePredicate:  # because we need non-local object, lambda is not ok
    threshold: float = 0.0

    def __call__(self, problem: TabletopClutteredFridgeReachingTask) -> bool:
        assert len(problem.descriptions) == 1
        desc = problem.descriptions[0]
        pos = desc[0].worldpos()
        is_y_positive = pos[1] > self.threshold
        return is_y_positive
