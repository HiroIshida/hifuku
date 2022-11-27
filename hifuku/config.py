from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, List, Tuple

import yaml


class GlobalConfig:
    hostport_pairs: Final[List[Tuple[str, int]]]

    def __init__(self, yaml_like=None):
        if isinstance(yaml_like, Mapping):
            conf = yaml_like
        else:
            if isinstance(yaml_like, Path):
                yaml_path = yaml_like
            elif yaml_like is None:
                yaml_path = Path("~/.config/hifuku.yaml").expanduser()
            else:
                assert False
            assert yaml_path.exists()
            with yaml_path.open(mode="r") as f:
                conf = yaml.safe_load(f)

        pairs = []
        if "hostport_pairs" in conf:
            hostport_pairs = conf["hostport_pairs"]
            assert isinstance(hostport_pairs, Sequence)
            assert not isinstance(hostport_pairs, str)
            for host, port in hostport_pairs:
                pair = (str(host), int(port))
                pairs.append(pair)
        self.hostport_pairs = pairs


global_config = GlobalConfig()
