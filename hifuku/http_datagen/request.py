import contextlib
import logging
import pickle
from dataclasses import asdict, dataclass
from http.client import HTTPConnection
from typing import List, Optional, Type, overload

import numpy as np

from hifuku.types import ProblemInterface

logger = logging.getLogger(__name__)


class Request:
    pass


class Response:
    pass


@dataclass
class GetCPUInfoRequest(Request):
    pass


@dataclass
class GetCPUInfoResponse(Response):
    n_cpu: int


@dataclass
class GetModuleHashValueRequest(Request):
    module_names: List[str]


@dataclass
class GetModuleHashValueResponse(Response):
    hash_values: List[Optional[str]]


@dataclass
class CreateDatasetRequest(Request):
    problem_type: Type[ProblemInterface]
    init_solution: np.ndarray
    n_problem: int
    n_problem_inner: int
    n_process: int

    def __str__(self) -> str:
        vis_dict = asdict(self)
        vis_dict["init_solution"] = "[{}..(float)..{}]".format(
            self.init_solution[0], self.init_solution[-1]
        )
        return vis_dict.__str__()


@dataclass
class CreateDatasetResponse(Response):
    data_list: List[bytes]
    name_list: List[str]
    elapsed_time: float

    def __str__(self) -> str:
        vis_dict = {}
        vis_dict["data_list"] = "[..({} binaries)..]".format(len(self.data_list))
        vis_dict["name_list"] = "[{}...{}".format(self.name_list[0], self.name_list[-1])
        vis_dict["elapsed_time"] = self.elapsed_time  # type: ignore
        return vis_dict.__str__()


@overload
def send_request(conn: HTTPConnection, request: GetCPUInfoRequest) -> GetCPUInfoResponse:
    pass


@overload
def send_request(
    conn: HTTPConnection, request: GetModuleHashValueRequest
) -> GetModuleHashValueResponse:
    pass


@overload
def send_request(conn: HTTPConnection, request: CreateDatasetRequest) -> CreateDatasetResponse:
    pass


def send_request(conn: HTTPConnection, request):
    headers = {"Content-type": "application/json"}
    conn.request("POST", "/post", pickle.dumps(request), headers)
    logger.debug("send request to ({}, {}): {}".format(conn.host, conn.port, request))
    response = pickle.loads(conn.getresponse().read())
    logger.debug("got renpose from ({}, {}): {}".format(conn.host, conn.port, response))
    return response


@contextlib.contextmanager
def http_connection(host: str = "localhost", port: int = 8080):
    conn = HTTPConnection(host, port)
    yield conn
    conn.close()
