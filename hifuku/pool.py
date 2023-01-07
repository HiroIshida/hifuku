import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Iterator, List, Optional, Type, TypeVar, cast

import numpy as np

from hifuku.llazy.dataset import DatasetIterator, LazyDecomplessDataset
from hifuku.rpbench_wrap import PicklableSamplableBase, PicklableTaskBase

logger = logging.getLogger(__name__)

TaskT = TypeVar("TaskT", bound=PicklableTaskBase)
SamplableT = TypeVar("SamplableT", bound=PicklableSamplableBase)
OtherSamplableT = TypeVar("OtherSamplableT", bound=PicklableSamplableBase)
CachedTaskT = TypeVar("CachedTaskT", bound=PicklableTaskBase)  # TODO: rename to CacheTaskT
PoolT = TypeVar("PoolT", bound="PoolLike")
T = TypeVar("T")


class PoolLike(ABC):
    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def split(self: PoolT, n_split: int) -> List[PoolT]:
        ...

    @abstractmethod
    def parallelizable(self) -> bool:
        ...


class TypicalPoolMixin:
    def reset(self) -> None:
        pass

    def split(self: PoolT, n_split: int) -> List[PoolT]:  # type: ignore[misc]
        return [copy.deepcopy(self) for _ in range(n_split)]

    def parallelizable(self) -> bool:
        return True


@dataclass
class PredicatedPool(PoolLike, Iterator[Optional[SamplableT]]):
    samplable_type: Type[SamplableT]
    n_desc_inner: int


@dataclass
class Pool(PoolLike, Iterator[SamplableT]):
    samplable_type: Type[SamplableT]
    n_desc_inner: int

    @abstractmethod
    def make_predicated(
        self, predicate: Callable[[SamplableT], bool], max_trial_factor: int
    ) -> PredicatedPool[SamplableT]:
        pass

    def as_predicated(self) -> PredicatedPool[SamplableT]:
        return cast(PredicatedPool[SamplableT], self)


@dataclass
class TrivialPredicatedPool(TypicalPoolMixin, PredicatedPool[SamplableT]):
    predicate: Callable[[SamplableT], bool]
    max_trial_factor: int

    def __next__(self) -> Optional[SamplableT]:
        return self.samplable_type.predicated_sample(
            self.n_desc_inner, self.predicate, self.max_trial_factor
        )


@dataclass
class TrivialPool(TypicalPoolMixin, Pool[SamplableT]):
    def __next__(self) -> SamplableT:
        return self.samplable_type.sample(self.n_desc_inner)

    def make_predicated(
        self, predicate: Callable[[SamplableT], bool], max_trial_factor: int
    ) -> TrivialPredicatedPool[SamplableT]:
        return TrivialPredicatedPool(
            self.samplable_type, self.n_desc_inner, predicate, max_trial_factor
        )


@dataclass
class PseudoIteratorPool(TypicalPoolMixin, Pool[SamplableT]):
    iterator: Iterator[SamplableT]

    def __next__(self) -> SamplableT:
        return next(self.iterator)

    def make_predicated(
        self, predicate: Callable[[SamplableT], bool], max_trial_factor: int
    ) -> PredicatedPool[SamplableT]:
        raise NotImplementedError("under construction")

    def reset(self) -> None:
        pass

    def parallelizable(self) -> bool:
        return True


@dataclass
class CachedPool(Pool[SamplableT], Generic[OtherSamplableT, SamplableT]):
    """Use Cache of precomputed SamplableT, create pool
    of SamplableT.

    This pool is beneficial specifically when sampling the world (including gridsdf)
    is not easy. In that case, if you have cache of other related tasks which can
    share the world, but not world-conditioned description, then you can use that
    cache as part of the task that you want to generate.
    """

    cache_path_list: List[Path]
    cache_samplable_type: Type[OtherSamplableT]
    dataset_iter: Optional[DatasetIterator] = None

    def __post_init__(self):
        assert len(self.cache_path_list) > 0
        assert self.dataset_iter is None, "maybe reset is not called yet?"

    def reset(self):
        dataset = LazyDecomplessDataset[OtherSamplableT](
            self.cache_path_list, self.cache_samplable_type, 1
        )
        self.dataset_iter = DatasetIterator(dataset)

    def split(self, n_split: int) -> List["CachedPool[OtherSamplableT, SamplableT]"]:
        indices_list = np.array_split(np.arange(len(self.cache_path_list)), n_split)
        pools = []
        for indices in indices_list:
            paths = [self.cache_path_list[i] for i in indices]
            pool = CachedPool(
                self.samplable_type, self.n_desc_inner, paths, self.cache_samplable_type, None
            )
            pools.append(pool)
        return pools

    def __next__(self) -> SamplableT:
        assert self.dataset_iter is not None
        data = next(self.dataset_iter)
        samplable = self.samplable_type.cast_from(data)
        if samplable.n_inner_task > 0:
            assert samplable.n_inner_task == self.n_desc_inner
            return samplable
        elif samplable.n_inner_task == 0:
            descs = samplable.sample_descriptions(samplable.world, self.n_desc_inner)
            return self.samplable_type(samplable.world, descs, samplable._gridsdf)
        else:
            assert False

    @classmethod
    def load(
        cls,
        cached_samplable_type: Type[OtherSamplableT],
        samplable_type: Type[SamplableT],
        n_desc_inner: int,
        cache_dir_path: Path,
    ):
        cache_path_list = [p for p in cache_dir_path.iterdir() if p.name.endswith(".gz")]
        return cls(samplable_type, n_desc_inner, cache_path_list, cached_samplable_type)

    def make_predicated(
        self, predicate: Callable[[SamplableT], bool], max_trial_factor: int
    ) -> PredicatedPool[SamplableT]:
        raise NotImplementedError("under construction")

    def parallelizable(self) -> bool:
        return False
