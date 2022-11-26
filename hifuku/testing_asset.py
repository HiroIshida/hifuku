# this file implements some class for testing

from dataclasses import dataclass

from hifuku.threedim.tabletop import TabletopPlanningProblem


@dataclass
class SimplePredicate:  # because we need non-local object, lambda is not ok
    threshold: float = 0.0

    def __call__(self, problem: TabletopPlanningProblem) -> bool:
        assert problem.n_problem() == 1
        pose = problem.target_pose_list[0]
        is_y_positive = pose.worldpos()[1] > self.threshold
        return is_y_positive
