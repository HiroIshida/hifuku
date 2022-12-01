import contextlib
import logging
import pickle
import time
from dataclasses import asdict, dataclass
from http.client import HTTPConnection
from typing import Generic, List, Optional, Tuple, TypeVar, overload

import numpy as np

from hifuku.pool import PredicatedProblemPool
from hifuku.types import ProblemT, ResultProtocol

logger = logging.getLogger(__name__)


class Request:
    ...


class Response:
    ...


class MainRequest(Request):
    n_process: int


class MainResponse(Response):
    elapsed_time: float


RequestT = TypeVar("RequestT", bound=Request)
ResponseT = TypeVar("ResponseT", bound=Response)
MainRequestT = TypeVar("MainRequestT", bound=MainRequest)
MainResponseT = TypeVar("MainResponseT", bound=MainResponse)


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
class SolveProblemRequest(Generic[ProblemT], MainRequest):
    problems: List[ProblemT]
    init_solutions: List[np.ndarray]
    n_process: int

    def __str__(self) -> str:
        return "[...hogehoge...]"


@dataclass
class SolveProblemResponse(MainResponse):
    results_list: List[Tuple[ResultProtocol, ...]]
    elapsed_time: float

    def __str__(self) -> str:
        return "[...hogehoge...]"


@dataclass
class SampleProblemRequest(Generic[ProblemT], MainRequest):
    n_sample: int
    pool: PredicatedProblemPool[ProblemT]
    n_process: int


@dataclass
class SampleProblemResponse(Generic[ProblemT], MainResponse):
    problems: List[ProblemT]
    elapsed_time: float

    def __str__(self) -> str:
        vis_dict = asdict(self)
        n_problems = len(self.problems)
        if n_problems > 0:
            vis_dict["problems"] = "[...({} problems)...]".format(n_problems)
        return vis_dict.__str__()


@overload
def send_request(conn: HTTPConnection, request: GetCPUInfoRequest) -> GetCPUInfoResponse:
    ...


@overload
def send_request(
    conn: HTTPConnection, request: GetModuleHashValueRequest
) -> GetModuleHashValueResponse:
    ...


@overload
def send_request(conn: HTTPConnection, request: SolveProblemRequest) -> SolveProblemResponse:
    ...


@overload
def send_request(conn: HTTPConnection, request: SampleProblemRequest) -> SampleProblemResponse:
    ...


@overload
def send_request(conn: HTTPConnection, request: MainRequest) -> MainResponse:
    ...


def send_request(conn: HTTPConnection, request):
    headers = {"Content-type": "application/json"}

    ts = time.time()
    serialized = pickle.dumps(request)
    logger.debug("elapsed time to serialize: {}".format(time.time() - ts))
    logger.debug("serialized object sizes: {} byte".format(len(serialized)))

    ts = time.time()
    conn.request("POST", "/post", serialized, headers)
    logger.debug("send request to ({}, {}): {}".format(conn.host, conn.port, request))
    logger.debug("elapsed time to send request: {}".format(time.time() - ts))

    raw_response = conn.getresponse().read()
    logger.debug("got renpose from ({}, {})".format(conn.host, conn.port))
    logger.debug("raw response object sizes: {} byte".format(len(raw_response)))

    ts = time.time()
    response = pickle.loads(raw_response)
    logger.debug("elapsed time to deserialize: {}".format(time.time() - ts))

    logger.debug("response contents: {}".format(response))
    return response


@contextlib.contextmanager
def http_connection(host: str = "localhost", port: int = 8080):
    logger.debug("try to connect {}".format((host, port)))
    conn = HTTPConnection(host, port)
    logger.debug("connected to {}!".format((host, port)))
    yield conn
    conn.close()
    logger.debug("closed connectionto {}!".format((host, port)))
