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
    def generate_single_data(self) -> ChunkT:
        pass

    def run(self) -> None:
        if self.arg.cpu_core is not None:
            cores = ",".join([str(e) for e in self.arg.cpu_core])
            os.system("taskset -p -c {} {}".format(cores, os.getpid()))
        unique_id = (uuid.getnode() + os.getpid()) % (2**32 - 1)
        np.random.seed(unique_id)
        disable_tqdm = not self.arg.show_process_bar

        for _ in tqdm.tqdm(range(self.arg.number), disable=disable_tqdm):
            chunk = self.generate_single_data()
            name = str(uuid.uuid4()) + self.arg.extension
            file_path = self.arg.base_path / name
            dump_path = chunk.dump(file_path)
            if self.arg.queue is not None:
                self.arg.queue.put(dump_path)
