import ast
import inspect
import pickle
import shutil
import subprocess
import tempfile
from hashlib import md5
from pathlib import Path
from typing import Any, Optional


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


if __name__ == "__main__":
    hash_value = get_module_source_hash("skrobot")
    print(hash_value)
