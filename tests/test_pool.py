import numpy as np
import pytest

from hifuku.pool import (
    IteratorProblemPool,
    SimpleFixedProblemPool,
    SimpleIteratorProblemPool,
)
from hifuku.threedim.tabletop import TabletopPlanningProblem


def test_simple_pool():
    pool = SimpleIteratorProblemPool(TabletopPlanningProblem, 10)

    for _ in range(5):
        next(pool)

    pred1 = lambda p: p.get_descriptions()[0][1] > -np.inf  # noqa
    pool_pred = pool.make_predicated(pred1, 40)
    for _ in range(5):
        assert next(pool_pred) is not None

    pred2 = lambda p: p.get_descriptions()[0][1] > np.inf  # noqa
    pool_pred = pool.make_predicated(pred2, 40)
    for _ in range(5):
        assert next(pool_pred) is None


def test_simple_fixed_pool():
    n_sample = 20
    fixed_pool = SimpleFixedProblemPool.initialize(TabletopPlanningProblem, n_sample)
    assert len(fixed_pool) == n_sample
    iterator_pool = fixed_pool.as_iterator()

    def infinite_loop(pool: IteratorProblemPool):
        while True:
            next(pool)

    with pytest.raises(StopIteration):
        infinite_loop(iterator_pool)
