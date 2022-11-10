from llazy.dataset import LazyDecomplessDataLoader, LazyDecomplessDataset
from mohou.file import get_project_path

from hifuku.types import RawData

pp = get_project_path("tabletop_ik")
chunk_dir_path = pp / "chunk"


dataset = LazyDecomplessDataset.load(chunk_dir_path, RawData)
dataset_train, dataset_valid = dataset.random_split(0.1)
train_loader = LazyDecomplessDataLoader(dataset_train, batch_size=200, max_data_num=5000)

from pyinstrument import Profiler

profiler = Profiler()
profiler.start()

for sample in train_loader:
    pass

profiler.stop()
print(profiler.output_text(unicode=True, color=True, show_all=True))
