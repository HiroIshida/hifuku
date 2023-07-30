import ast
import contextlib
import inspect
import logging
import os
import pickle
import platform
import shutil
import subprocess
import tempfile
import time
import warnings
from datetime import datetime
from hashlib import md5
from logging import Logger
from pathlib import Path
from typing import Any, List, Optional

import torch


def detect_physical_cpu_num() -> int:
    n_cpu = os.cpu_count()
    assert n_cpu is not None
    if platform.machine() == "aarch64":
        return n_cpu - 2  # 2 is a magick number
    elif platform.machine() == "x86_64":
        # because in hyperthreading, chache is shared.
        # so we must count only the physical cores
        # NOTE: hyperthreading can actully be off in bios though...
        return int(n_cpu * 0.5)
    else:
        assert "please implement for platform {}".format(platform.machine())
    assert False


def get_random_seed() -> int:
    unique_seed = datetime.now().microsecond + os.getpid()
    return unique_seed


@contextlib.contextmanager
def num_torch_thread(n_thread: int):
    n_thread_original = torch.get_num_threads()
    torch.set_num_threads(n_thread)
    yield
    torch.set_num_threads(n_thread_original)


def get_source_code_hash(any: Any) -> str:
    source_code = inspect.getsource(any)
    tree = ast.parse(source_code)
    hash_value = md5(pickle.dumps(tree)).hexdigest()
    return hash_value


def get_module_source_hash(module_name: str) -> Optional[str]:
    try:
        exec("import {}".format(module_name))
    except ImportError:
        return None

    file_path_str = eval("{}.__file__".format(module_name))
    package_directory_path = Path(file_path_str).parent

    Path(file_path_str).parent
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        copied_module_path = td_path / module_name
        shutil.copytree(package_directory_path, copied_module_path)
        cmd = "cd {0} && tar cf tmp.tar $(find -name '*.py') && md5sum tmp.tar".format(td_path)
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
        hash_value = proc.stdout.decode("utf8").split()[0]
    return hash_value


def create_default_logger(project_path: Path, prefix: str) -> Logger:
    timestr = "_" + time.strftime("%Y%m%d%H%M%S")
    log_dir_path = project_path / "log"
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir_path / (prefix + timestr + ".log")

    logger = logging.getLogger()  # root logger
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(str(log_file_path))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)

    log_sym_path = log_dir_path / ("latest_" + prefix + ".log")

    logger.info("create log symlink :{0} => {1}".format(log_file_path, log_sym_path))
    if log_sym_path.is_symlink():
        log_sym_path.unlink()
    log_sym_path.symlink_to(log_file_path)

    return logger


def filter_warnings():
    warnings.filterwarnings("ignore", message="Values in x were outside bounds during")
    warnings.filterwarnings("ignore", message="texture specified in URDF is not supported")
    warnings.filterwarnings("ignore", message="Converting sparse A to a CSC")
    warnings.filterwarnings("ignore", message="urllib3")
    warnings.filterwarnings(
        "ignore",
        message="undefined symbol: _ZNK3c1010TensorImpl36is_contiguous_nondefault_policy_implENS_12MemoryFormatE",
    )


def split_number(n: int, k: int) -> List[int]:
    quotient, remainder = divmod(n, k)
    return [quotient + (i < remainder) for i in range(k)]


warnings.filterwarnings(
    "ignore", message="`np.float` is a deprecated alias for the builtin `float`"
)


if __name__ == "__main__":
    hash_value = get_module_source_hash("skrobot")
    print(hash_value)
