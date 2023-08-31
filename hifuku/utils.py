import ast
import contextlib
import inspect
import os
import pickle
import platform
import shutil
import subprocess
import tempfile
import warnings
from datetime import datetime
from hashlib import md5
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch


def determine_process_thread() -> Tuple[int, int]:
    n_cpu = os.cpu_count()
    assert n_cpu is not None
    if platform.machine() == "aarch64":
        n_process = n_cpu - 2  # 2 is a magick number
        n_thread = 1
        return n_process, n_thread
    elif platform.machine() == "x86_64":
        # because in hyperthreading, chache is shared.
        # so we must count only the physical cores
        # NOTE: hyperthreading can actully be off in bios though...
        n_process = int(n_cpu * 0.5)
        n_thread = 1
        return n_process, n_thread
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


def split_number(n: int, k: int) -> List[int]:
    quotient, remainder = divmod(n, k)
    return [quotient + (i < remainder) for i in range(k)]


warnings.filterwarnings(
    "ignore", message="`np.float` is a deprecated alias for the builtin `float`"
)


if __name__ == "__main__":
    hash_value = get_module_source_hash("skrobot")
    print(hash_value)
