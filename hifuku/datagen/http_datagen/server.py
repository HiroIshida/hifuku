import argparse
import logging
import multiprocessing
import pickle
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from skmp.solver.interface import ConfigT, ResultT

from hifuku.datagen import (
    MultiProcesBatchMarginsDeterminant,
    MultiProcessBatchProblemSampler,
    MultiProcessBatchProblemSolver,
)
from hifuku.datagen.http_datagen.request import (
    DetermineMarginsRequest,
    DetermineMarginsResponse,
    GetCPUInfoRequest,
    GetCPUInfoResponse,
    GetModuleHashValueRequest,
    GetModuleHashValueResponse,
    Response,
    SampleProblemRequest,
    SampleProblemResponse,
    SolveProblemRequest,
    SolveProblemResponse,
)
from hifuku.pool import ProblemT
from hifuku.script_utils import watch_memmory
from hifuku.utils import determine_process_thread, get_module_source_hash


def split_number(num, div):
    return [num // div + (1 if x < num % div else 0) for x in range(div)]


class PostHandler(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

    def process_GetCPUInfoRequest(self, request: GetCPUInfoRequest) -> GetCPUInfoResponse:
        n_process, _ = determine_process_thread()
        logging.info("available process count: {}".format(n_process))
        resp = GetCPUInfoResponse(n_process)
        return resp

    def process_GetModuleHashValueRequest(
        self, request: GetModuleHashValueRequest
    ) -> GetModuleHashValueResponse:
        hashes = [get_module_source_hash(name) for name in request.module_names]
        resp = GetModuleHashValueResponse(hashes)
        return resp

    def process_SolveProblemRequest(
        self, request: SolveProblemRequest[ProblemT, ConfigT, ResultT]
    ) -> SolveProblemResponse[ResultT]:
        ts = time.time()
        logging.info("request: {}".format(request))

        gen = MultiProcessBatchProblemSolver[ConfigT, ResultT](
            request.solver_t, request.config, request.n_process
        )
        results_list = gen.solve_batch(
            request.problems, request.init_solutions, request.use_default_solver
        )

        elapsed_time = time.time() - ts
        resp = SolveProblemResponse(results_list, elapsed_time)
        return resp

    def process_SampleProblemRequest(
        self, request: SampleProblemRequest[ProblemT]
    ) -> SampleProblemResponse[ProblemT]:

        # NOTE: by calling this line (sample()) some pre-computation
        # e.g. sdf mesh creation will running.
        # Without this line, all the processes will do pre-computation
        # by themself in MultiProcessBatchProblemSampler which sometimes
        # stall the entire procedure
        request.pool.problem_type.sample(1, standard=True)  # don't delete

        ts = time.time()
        logging.info("request: {}".format(request))
        sampler = MultiProcessBatchProblemSampler[ProblemT](request.n_process)
        problems = sampler.sample_batch(request.n_sample, request.pool, request.delete_cache)
        assert len(problems) > 0
        elapsed_time = time.time() - ts
        resp = SampleProblemResponse(problems, elapsed_time)
        return resp

    def process_DetermineMarginsRequest(
        self, request: DetermineMarginsRequest
    ) -> DetermineMarginsResponse:
        ts = time.time()
        logging.info("request: {}".format(request))
        determinant = MultiProcesBatchMarginsDeterminant(request.n_process)
        results = determinant.determine_batch(
            request.n_sample,
            request.coverage_results,
            request.threshold,
            request.target_fp_rate,
            request.cma_sigma,
            request.margins_guess,
            request.minimum_coverage,
        )
        elapsed_time = time.time() - ts
        resp = DetermineMarginsResponse(results, elapsed_time)
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
        elif isinstance(request, SolveProblemRequest):
            resp = self.process_SolveProblemRequest(request)
        elif isinstance(request, SampleProblemRequest):
            resp = self.process_SampleProblemRequest(request)
        elif isinstance(request, DetermineMarginsRequest):
            resp = self.process_DetermineMarginsRequest(request)
        else:
            assert False, "request {} is not supported".format(type(request))
        print("time to process: {}".format(time.time() - ts))
        self.wfile.write(pickle.dumps(resp))
        print("elapsed time to handle request (including sending): {}".format(time.time() - ts))


def run_server(server_class=HTTPServer, port=8080, with_memory_watchdog: bool = False):
    logging.basicConfig(level=logging.INFO)
    server_address = ("", port)
    httpd = server_class(server_address, PostHandler)

    if with_memory_watchdog:
        p_watchdog = multiprocessing.Process(target=watch_memmory, args=(5.0, False))
        p_watchdog.start()

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
    parser.add_argument("--wm", action="store_true", help="watch memory")
    args = parser.parse_args()
    run_server(port=args.port, with_memory_watchdog=args.wm)
