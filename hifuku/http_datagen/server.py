import argparse
import logging
import os
import pickle
import tempfile
import time
from abc import ABC, abstractmethod
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Type

from hifuku.datagen import MultiProcessDatasetGenerator
from hifuku.http_datagen.request import (
    CreateDatasetRequest,
    CreateDatasetResponse,
    GetCPUInfoRequest,
    GetCPUInfoResponse,
    GetModuleHashValueRequest,
    GetModuleHashValueResponse,
    Response,
)
from hifuku.llazy.generation import DataGenerationTask
from hifuku.utils import get_module_source_hash


def split_number(num, div):
    return [num // div + (1 if x < num % div else 0) for x in range(div)]


class PostHandler(BaseHTTPRequestHandler, ABC):
    @classmethod
    @abstractmethod
    def get_task_type(cls) -> Type[DataGenerationTask]:
        ...

    def _set_response(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

    def process_GetCPUInfoRequest(self, request: GetCPUInfoRequest) -> GetCPUInfoResponse:
        n_cpu = os.cpu_count()
        assert n_cpu is not None
        cpu_count = int(n_cpu * 0.5)
        logging.info("cpu count: {}".format(cpu_count))
        resp = GetCPUInfoResponse(cpu_count)
        return resp

    def process_GetModuleHashValueRequest(
        self, request: GetModuleHashValueRequest
    ) -> GetModuleHashValueResponse:
        hashes = [get_module_source_hash(name) for name in request.module_names]
        resp = GetModuleHashValueResponse(hashes)
        return resp

    def process_CreateDatasetRequest(self, request: CreateDatasetRequest) -> CreateDatasetResponse:
        ts = time.time()
        logging.info("request: {}".format(request))

        gen = MultiProcessDatasetGenerator(request.problem_type, request.n_process)

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            gen.generate(request.init_solution, request.n_problem, request.n_problem_inner, td_path)

            byte_data_list = []
            name_list = []
            for path in td_path.iterdir():
                with path.open(mode="rb") as f:
                    byte_data = f.read()
                    byte_data_list.append(byte_data)
                    name_list.append(path.name)

        elapsed_time = time.time() - ts
        resp = CreateDatasetResponse(byte_data_list, name_list, elapsed_time)
        return resp

    def do_POST(self):
        ts = time.time()

        content_length = int(self.headers["Content-Length"])
        request = pickle.loads(self.rfile.read(content_length))
        logging.info("recieved request type: {}".format(type(request)))

        self._set_response()

        resp: Response
        if isinstance(request, GetCPUInfoRequest):
            resp = self.process_GetCPUInfoRequest(request)
        elif isinstance(request, GetModuleHashValueRequest):
            resp = self.process_GetModuleHashValueRequest(request)
        elif isinstance(request, CreateDatasetRequest):
            resp = self.process_CreateDatasetRequest(request)
        else:
            assert False
        self.wfile.write(pickle.dumps(resp))
        print("elapsed time to handle request: {}".format(time.time() - ts))


def run_server(task_type: Type[DataGenerationTask], server_class=HTTPServer, port=8080):
    class CustomHandler(PostHandler):
        @classmethod
        def get_task_type(cls) -> Type[DataGenerationTask]:
            return task_type

    logging.basicConfig(level=logging.INFO)
    server_address = ("", port)
    httpd = server_class(server_address, CustomHandler)
    logging.info("Starting httpd...\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info("Stopping httpd...\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-port", type=int, default=8080, help="port number")
    args = parser.parse_args()
    port: int = args.port
    # run_server(PostHandler, port=port)
