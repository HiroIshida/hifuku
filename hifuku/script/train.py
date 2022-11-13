from mohou.file import get_project_path
from mohou.script_utils import create_default_logger
from mohou.trainer import TrainCache, TrainConfig, train

from hifuku.nerual import (
    IterationPredictor,
    IterationPredictorConfig,
    IterationPredictorDataset,
    VoxelAutoEncoder,
)

pp = get_project_path("tabletop_ik")
chunk_dir_path = pp / "chunk"


ae_model = TrainCache.load(pp, VoxelAutoEncoder).best_model
dataset = IterationPredictorDataset.load(chunk_dir_path, ae_model)

# dataset = LazyDecomplessDataset.load(chunk_dir_path, RawData, n_worker=-1)
# dataset_train, dataset_valid = dataset.random_split(0.1)
# train_loader = LazyDecomplessDataLoader(dataset_train, batch_size=500)
# valid_loader = LazyDecomplessDataLoader(dataset_valid, batch_size=50, shuffle=False)

logger = create_default_logger(pp, "iteration_predictor")
model_conf = IterationPredictorConfig(12, ae_model.config.dim_bottleneck, 10)
tcache = TrainCache.from_model(IterationPredictor(model_conf))
train_conf = TrainConfig(n_epoch=100, batch_size=100)
train(pp, tcache, dataset, train_conf)
