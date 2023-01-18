import numpy as np
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skmp.trajectory import Trajectory

from hifuku.http_datagen.request import http_connection, send_request
from hifuku.http_datagen.server import (
    GetCPUInfoRequest,
    GetModuleHashValueRequest,
    SolveProblemRequest,
)
from hifuku.rpbench_wrap import TabletopBoxRightArmReachingTask

with http_connection("localhost", 8081) as conn:
    req1 = GetModuleHashValueRequest(["skrobot", "tinyfk", "skplan"])
    try:
        resp1 = send_request(conn, req1)
        print(resp1.hash_values)
    except Exception as e:
        print(e)
        print(type(e))


with http_connection("localhost", 8080) as conn:
    req1 = GetModuleHashValueRequest(["skrobot", "tinyfk", "skplan"])
    resp1 = send_request(conn, req1)
    print(resp1.hash_values)

    req2 = GetCPUInfoRequest()
    resp2 = send_request(conn, req2)

    problems = [TabletopBoxRightArmReachingTask.sample(2) for _ in range(3)]
    init_solution = Trajectory(list(np.zeros((10, 15))))
    init_solutions = [init_solution] * 3
    req3 = SolveProblemRequest(
        problems, OMPLSolver, OMPLSolverConfig(), init_solutions, resp2.n_cpu
    )
    resp3 = send_request(conn, req3)
    print(resp3.elapsed_time)
