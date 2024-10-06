import ctypes
from os import path
from pathlib import Path

import numpy as np

this_source_path = Path(path.abspath(__file__))
lib_path = this_source_path.parent / "compute_coverage_and_fp.so"
if not lib_path.exists():
    import subprocess

    # use c++17
    cmd = f"g++ -shared -fPIC -std=c++17 -O3 -o {lib_path} {this_source_path.parent}/compute_coverage_and_fp.cpp"
    ret = subprocess.run(cmd, shell=True)
    assert ret.returncode == 0

lib = ctypes.CDLL(str(lib_path))
lib.compute_coverage_and_fp_jit.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),  # biases
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),  # realss
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),  # estss
    ctypes.c_double,  # threshold
    ctypes.c_int,  # N_path
    ctypes.c_int,  # N_mc
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),  # optimal_coverage
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),  # optimal_fp
]
lib.compute_coverage_and_fp_jit.restype = None
compute_coverage_and_fp_cpp = lib.compute_coverage_and_fp_jit
