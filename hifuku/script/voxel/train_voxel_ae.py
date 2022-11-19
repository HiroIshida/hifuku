from mohou.file import get_project_path
from mohou.script_utils import create_default_logger
from mohou.trainer import TrainCache, TrainConfig, train_lower

from hifuku.llazy.dataset import LazyDecomplessDataLoader, LazyDecomplessDataset
from hifuku.neuralnet import VoxelAutoEncoder, VoxelAutoEncoderConfig
from hifuku.types import RawData

pp = get_project_path("tabletop_mesh")
cache_dir_path = pp / "cache"

logger = create_default_logger(pp, "voxel_ae")

dataset = LazyDecomplessDataset.load(cache_dir_path, RawData, n_worker=-1)
dataset_train, dataset_valid = dataset.random_split(0.1)
train_loader = LazyDecomplessDataLoader(dataset_train, batch_size=500)
valid_loader = LazyDecomplessDataLoader(dataset_valid, batch_size=50, shuffle=False)

model_conf = VoxelAutoEncoderConfig()
tcache = TrainCache.from_model(VoxelAutoEncoder(model_conf))
train_conf = TrainConfig(n_epoch=50)
train_lower(pp, tcache, train_loader, valid_loader, train_conf)
