import time
from pathlib import Path

import grpc

import hifuku.distributed.datagen_pb2 as datagen_pb2
import hifuku.distributed.datagen_pb2_grpc as datagen_pb2_grpc


def make_request_and_redirect(n_data: int, redirect_path: Path, host: str = "localhost:5051"):

    with grpc.insecure_channel("localhost:5051") as channel:
        stub = datagen_pb2_grpc.DataGenServiceStub(channel)
        req = datagen_pb2.DataGenRequest(n_data=n_data)
        response = stub.DataGenStream(req)
        time.sleep(1)

        for resp in response:
            buf, file_name = resp
            print(file_name)
            file_path: Path = redirect_path / file_name
            with file_path.open(mode="wb") as f:
                f.write(buf)
