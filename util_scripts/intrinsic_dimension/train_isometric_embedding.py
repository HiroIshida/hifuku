import pickle
from pathlib import Path

from mohou.trainer import TrainConfig
from quotdim.neural import QuotDimDataset, train_for_different_embedding_dimension

if __name__ == "__main__":
    with Path("./dataset.pkl").open(mode="rb") as f:
        raw_dataset = pickle.load(f)
    # raw_dataset = [e for e in raw_dataset if e[2] < 500]

    dataset = QuotDimDataset(raw_dataset)
    print("n_sample: {}".format(len(dataset)))

    config = TrainConfig(learning_rate=0.005, n_epoch=3000)

    pp = Path("isometric_embedding/")
    train_for_different_embedding_dimension(dataset, [2, 3], 100, n_mc=1, config=config, pp=pp)
