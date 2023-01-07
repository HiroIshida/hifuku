import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from skrobot.coordinates import Coordinates

from hifuku.pool import CachedPool, TrivialPool
from hifuku.rpbench_wrap import TabletopBoxRightArmReachingTask, TabletopBoxWorldWrap


def test_simple_pool():
    pool = TrivialPool(TabletopBoxRightArmReachingTask, 10)

    for _ in range(5):
        next(pool)

    pool_pred = pool.make_predicated(lambda x: True, 40)
    for _ in range(5):
        assert next(pool_pred) is not None

    pool_pred = pool.make_predicated(lambda x: False, 40)
    for _ in range(5):
        assert next(pool_pred) is None


def test_cached_pool():
    with TemporaryDirectory() as td:
        cache_path = Path(td)
        for _ in range(20):
            file_path = cache_path / str(uuid.uuid4())
            ww = TabletopBoxWorldWrap.sample(0)
            ww.dump(file_path)

        n_inner = 10
        pool = CachedPool.load(
            TabletopBoxWorldWrap, TabletopBoxRightArmReachingTask, n_inner, cache_path
        )
        pool.reset()
        for _ in range(20):
            task = next(pool)
            assert isinstance(task.descriptions[0][0], Coordinates)
            assert len(task.descriptions) == n_inner
