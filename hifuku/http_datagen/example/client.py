from hifuku.http_datagen.request import http_connection, send_request
from hifuku.http_datagen.server import (
    CreateDatasetRequest,
    GetCPUInfoRequest,
    GetModuleHashValueRequest,
)

with http_connection("localhost", 8080) as conn:
    req1 = GetModuleHashValueRequest(["skrobot", "tinyfk", "skplan"])
    resp1 = send_request(conn, req1)
    print(resp1.hash_values)

    req2 = GetCPUInfoRequest()
    resp2 = send_request(conn, req2)

    req3 = CreateDatasetRequest(84, resp2.n_cpu)
    resp3 = send_request(conn, req3)
    print(resp3.elapsed_time)
