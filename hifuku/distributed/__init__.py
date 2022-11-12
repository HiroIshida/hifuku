import subprocess
from pathlib import Path

from grpc_tools import protoc

try:
    this_dir_path = Path(__file__).parent
    assert (this_dir_path / "datagen_pb2_grpc.py").exists()
    assert (this_dir_path / "datagen_pb2.py").exists()
except:
    this_dir_path = Path(__file__).parent
    proto_path = this_dir_path / "datagen.proto"
    out_path = this_dir_path
    command = (
        "",
        "-I{}".format(this_dir_path),
        "--python_out={}".format(out_path),
        "--grpc_python_out={}".format(out_path),
        str(proto_path),
    )
    protoc.main(command)
    # Too dirty
    cmd = "sed -i 's/import datagen_pb2 as datagen__pb2/import hifuku.distributed.datagen_pb2 as datagen__pb2/g' {}/datagen_pb2_grpc.py".format(
        this_dir_path
    )
    subprocess.run(cmd, shell=True)
