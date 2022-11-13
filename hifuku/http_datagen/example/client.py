from hifuku.http_datagen.request import send_request
from hifuku.http_datagen.server import (
    CreateDatasetRequest,
    GetCPUInfoRequest,
    GetModuleHashValueRequest,
)

req1 = GetModuleHashValueRequest(["skrobot", "tinyfk", "skplan"])
resp1 = send_request(req1, 8080)
print(resp1.hash_values)

req2 = GetCPUInfoRequest()
resp2 = send_request(req2, 8080)

req3 = CreateDatasetRequest(84, resp2.n_cpu)
resp3 = send_request(req3, 8080)
print(resp3.elapsed_time)
