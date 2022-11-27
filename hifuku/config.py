from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import yaml


@dataclass
class ServerSpec:
    name: str
    port: int
    perf: float


@dataclass(frozen=True)
class GlobalConfig:
    server_specs: Tuple[ServerSpec, ...]

    @classmethod
    def load(cls, yaml_like=None):
        if isinstance(yaml_like, Mapping):
            conf = yaml_like
        else:
            if isinstance(yaml_like, Path):
                yaml_path = yaml_like
            elif yaml_like is None:
                yaml_path = Path("~/.config/hifuku.yaml").expanduser()
            else:
                assert False

            if yaml_path.exists():
                with yaml_path.open(mode="r") as f:
                    conf = yaml.safe_load(f)
            else:
                conf = {}

        specs: List[ServerSpec] = []
        if "server" in conf:
            for spec_dict in conf["server"]:
                spec = ServerSpec(
                    spec_dict["name"], int(spec_dict["port"]), float(spec_dict["perf"])
                )
                specs.append(spec)

        if len(specs) > 0:
            perf_total = sum(spec.perf for spec in specs)
            for spec in specs:
                spec.perf /= perf_total

        return cls(tuple(specs))


global_config = GlobalConfig.load()
