import logging
import multiprocessing
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Dict, Generic, List, Optional

import numpy as np
import tqdm

from hifuku.llazy.dataset import ChunkT
from hifuku.utils import get_random_seed

logger = logging.getLogger(__name__)


@dataclass
class DataGenerationTaskArg:
    number: int
    show_process_bar: bool
    base_path: Path
    info: Optional[Dict] = None
    cpu_core: Optional[List[int]] = None
    queue: Optional[multiprocessing.Queue] = None
    extension: str = ".pkl"


class DataGenerationTask(ABC, Process, Generic[ChunkT]):
    arg: DataGenerationTaskArg

    def __init__(self, arg: DataGenerationTaskArg):
        self.arg = arg
        super().__init__()
        self.post_init_hook()

    @abstractmethod
    def post_init_hook(self) -> None:
        pass

    @abstractmethod
    def generate_single_data(self) -> Optional[ChunkT]:
        pass

    def run(self) -> None:
        logger.debug("DataGenerationTask.run with pid {}".format(os.getpid()))

        if self.arg.cpu_core is not None:
            logger.debug("cpu core is specified => {}".format(self.arg.cpu_core))
            cores = ",".join([str(e) for e in self.arg.cpu_core])
            command = "taskset -p -c {} {}".format(cores, os.getpid())
            logger.debug("command => {}".format(command))
            os.system(command)

        random_seed = get_random_seed()
        logger.debug("random seed set to {}".format(random_seed))
        np.random.seed(random_seed)
        disable_tqdm = not self.arg.show_process_bar

        with tqdm.tqdm(total=self.arg.number, disable=disable_tqdm) as pbar:
            counter = 0
            while counter < self.arg.number:
                chunk = self.generate_single_data()
                if chunk is None:
                    continue
                name = str(uuid.uuid4()) + self.arg.extension
                file_path = self.arg.base_path / name
                dump_path = chunk.dump(file_path)
                if self.arg.queue is not None:
                    self.arg.queue.put(dump_path)
                pbar.update(1)
                counter += 1
