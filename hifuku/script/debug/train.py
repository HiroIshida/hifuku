import numpy as np
from mohou.file import get_project_path
from mohou.script_utils import create_default_logger
from mohou.trainer import TrainCache, TrainConfig, train

from hifuku.domain import TBRR_SQP_Domain
from hifuku.llazy.dataset import LazyDecomplessDataset
from hifuku.neuralnet import (
    IterationPredictor,
    IterationPredictorConfig,
    IterationPredictorDataset,
)
from hifuku.script_utils import load_compatible_autoencoder
from hifuku.types import RawData

ae_model = load_compatible_autoencoder(TBRR_SQP_Domain)

pp = get_project_path("TBRR_SQP")
chunk_dir_path = pp / "cache"

dataset = IterationPredictorDataset.load(chunk_dir_path, ae_model)
raw_dataset = LazyDecomplessDataset.load(chunk_dir_path, RawData, n_worker=-1)
rawdata = raw_dataset.get_data(np.array([0]))[0]
init_solution = rawdata.init_solution

logger = create_default_logger(pp, "iteration_predictor")
model_conf = IterationPredictorConfig(12, ae_model.n_bottleneck)
model = IterationPredictor(model_conf)
model.initial_solution = init_solution
tcache = TrainCache.from_model(model)
train_conf = TrainConfig(n_epoch=100, batch_size=100)
train(pp, tcache, dataset, train_conf)
