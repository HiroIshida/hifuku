import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

from mohou.trainer import TrainCache, TrainConfig, train

from hifuku.intrinsic_dimension.metricnet import (
    MetricNet,
    MetricNetConfig,
    MetricNetDataset,
)

if __name__ == "__main__":
    with Path("./dataset.pkl").open(mode="rb") as f:
        raw_dataset = pickle.load(f)
    raw_dataset = [e for e in raw_dataset if e[2] < 500]

    dataset = MetricNetDataset(raw_dataset)
    dim = len(dataset[0][0])

    with TemporaryDirectory() as td:
        train_config = TrainConfig(learning_rate=0.01, n_epoch=50)
        model = MetricNet(MetricNetConfig(dim))
        tcache = TrainCache.from_model(model)
        train(Path(td), tcache, dataset=dataset, config=train_config)

        hifuku_data_path = Path("hifuku_tcache.pkl")
        with hifuku_data_path.open(mode="wb") as f:
            pickle.dump(tcache, f)
