import multiprocessing
import os
import tempfile
import time
from concurrent import futures
from pathlib import Path
from typing import Type

import grpc

import hifuku.distributed.datagen_pb2 as datagen_pb2
import hifuku.distributed.datagen_pb2_grpc as datagen_pb2_grpc
from hifuku.llazy.generation import DataGenerationTask, DataGenerationTaskArg


def split_number(num, div):
    return [num // div + (1 if x < num % div else 0) for x in range(div)]


class Server(datagen_pb2_grpc.DataGenServiceServicer):
    task_type: Type[DataGenerationTask]
    stream_running_flag: bool

    def __init__(self, task_type: Type[DataGenerationTask]):
        self.task_type = task_type
        self.stream_running_flag = False

    def yield_generated_gz(self, q: multiprocessing.Queue):
        while True:
            print("waiting for getting queue")
            path = q.get()
            if path is None:
                break
            with path.open(mode="rb") as f:
                buf = f.read()
            resp = datagen_pb2.DataGenResponse(buf, path.name)
            yield resp

    def DataGenStream(self, request, context):
        if self.stream_running_flag:
            print("rejected")
            return

        print("recived")
        self.stream_running_flag = True

        n_data: int = request.n_data

        cpu_num = os.cpu_count()
        assert cpu_num is not None
        n_process = int(cpu_num * 0.5)
        assert n_process is not None
        n_data_list = split_number(n_data, n_process)
        print("n_worker {}".format(n_process))

        queue = multiprocessing.Queue()
        # thread_args = (queue,)
        # thread = threading.Thread(target=self.yield_generated_gz, args=thread_args)
        # thread.start()
        # print("start thread")

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            process_list = []
            for idx_process, n_data in enumerate(n_data_list):
                # show_process_bar = idx_process == 1
                show_process_bar = True
                arg = DataGenerationTaskArg(
                    n_data, show_process_bar, td_path, extension=".npz", queue=queue
                )
                p = self.task_type(arg)
                print("start process")
                p.start()
                process_list.append(p)

            for p in process_list:
                p.join()
        print("finish all processes")
        queue.put(None)  # put sentinel to finish thread
        self.stream_running_flag = False

        while True:
            print("hoge")
            path = queue.get()
            if path is None:
                break
            with path.open(mode="rb") as f:
                buf = f.read()
            resp = datagen_pb2.DataGenResponse(buf, path.name)
            yield resp

        self.stream_running_flag = False
        print("finish")


def run_server(task_type: Type[DataGenerationTask]):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    datagen_pb2_grpc.add_DataGenServiceServicer_to_server(Server(task_type), server)
    server.add_insecure_port("[::]:5051")
    server.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop(0)
