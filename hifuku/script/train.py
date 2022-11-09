from llazy.dataset import LazyDecomplessDataLoader, LazyDecomplessDataset
from mohou.file import get_project_path
from mohou.trainer import TrainCache, TrainConfig, train_lower

from hifuku.nerual import IterationPredictor, IterationPredictorConfig
from hifuku.threedim.tabletop import TabletopIKData

pp = get_project_path("tabletop_ik")
chunk_dir_path = pp / "chunk"

dataset = LazyDecomplessDataset.load(chunk_dir_path, TabletopIKData, n_worker=-1)
dataset_train, dataset_valid = dataset.random_split(0.1)
train_loader = LazyDecomplessDataLoader(dataset_train, batch_size=200)
valid_loader = LazyDecomplessDataLoader(dataset_valid, batch_size=200, shuffle=False)

model_conf = IterationPredictorConfig(6)
tcache = TrainCache.from_model(IterationPredictor(model_conf))
train_conf = TrainConfig(n_epoch=30)
train_lower(pp, tcache, train_loader, valid_loader, train_conf)
