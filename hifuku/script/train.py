import numpy as np
from mohou.file import get_project_path
from mohou.script_utils import create_default_logger
from mohou.trainer import TrainCache, TrainConfig, train

from hifuku.llazy.dataset import LazyDecomplessDataset
from hifuku.neuralnet import (
    IterationPredictor,
    IterationPredictorConfig,
    IterationPredictorDataset,
    VoxelAutoEncoder,
)
from hifuku.types import RawData

pp = get_project_path("tabletop_ik")
chunk_dir_path = pp / "chunk"

ae_model = TrainCache.load(pp, VoxelAutoEncoder).best_model
dataset = IterationPredictorDataset.load(chunk_dir_path, ae_model)
raw_dataset = LazyDecomplessDataset.load(chunk_dir_path, RawData, n_worker=-1)
rawdata = raw_dataset.get_data(np.array([0]))[0]
init_solution = rawdata.init_solution

logger = create_default_logger(pp, "iteration_predictor")
model_conf = IterationPredictorConfig(12, ae_model.config.dim_bottleneck, 10)
model = IterationPredictor(model_conf)
model.initial_solution = init_solution
tcache = TrainCache.from_model(model)
train_conf = TrainConfig(n_epoch=100, batch_size=100)
train(pp, tcache, dataset, train_conf)
