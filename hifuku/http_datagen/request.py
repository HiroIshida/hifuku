import contextlib
import pickle
from dataclasses import dataclass
from http.client import HTTPConnection
from typing import List, Optional, overload


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
    n_data: int
    n_process: int


@dataclass
class CreateDatasetResponse(Response):
    data_list: List[bytes]
    elapsed_time: float


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
    response = pickle.loads(conn.getresponse().read())
    return response


@contextlib.contextmanager
def http_connection(host: str = "localhost", port: int = 8080):
    conn = HTTPConnection(host, port)
    yield conn
    conn.close()
