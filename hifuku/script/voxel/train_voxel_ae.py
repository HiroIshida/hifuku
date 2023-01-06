from mohou.file import get_project_path
from mohou.script_utils import create_default_logger
from mohou.trainer import TrainCache, TrainConfig, train_lower

from hifuku.llazy.dataset import LazyDecomplessDataLoader, LazyDecomplessDataset
from hifuku.neuralnet import VoxelAutoEncoder, VoxelAutoEncoderConfig
from hifuku.task_wrap import TabletopBoxWorldWrap

problem_type = TabletopBoxWorldWrap

pp = get_project_path("tabletop_mesh-{}".format(problem_type.__name__))
cache_base_path = pp / "cache"
cache_base_path.mkdir(exist_ok=True, parents=True)

logger = create_default_logger(pp, "voxel_ae")

dataset = LazyDecomplessDataset.load(cache_base_path, problem_type, n_worker=-1)
print("finish setup dataset")
dataset_train, dataset_valid = dataset.random_split(0.1)
train_loader = LazyDecomplessDataLoader(dataset_train, batch_size=500)
valid_loader = LazyDecomplessDataLoader(dataset_valid, batch_size=50, shuffle=False)
print("finish setup loader")

model_conf = VoxelAutoEncoderConfig()
tcache = TrainCache.from_model(VoxelAutoEncoder(model_conf))
train_conf = TrainConfig(n_epoch=100)
train_lower(pp, tcache, train_loader, valid_loader, train_conf)
