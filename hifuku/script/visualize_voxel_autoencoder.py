from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mohou.trainer import TrainCache
from rpbench.articulated.pr2.minifridge import PR2MiniFridgeTask

from hifuku.neuralnet import VoxelAutoEncoder

if __name__ == "__main__":
    tcache = TrainCache.load_latest(Path("./voxel_autoencoder/"), VoxelAutoEncoder)
    task = PR2MiniFridgeTask.sample()
    vmap = task.world.create_voxelmap()
    vmap_torch = torch.from_numpy(vmap).unsqueeze(0).unsqueeze(0).float()

    model = tcache.best_model
    vmap_torch = model.decoder(model.encoder(vmap_torch))
    vmap_np = vmap_torch.squeeze().detach().numpy() > 0.5
    print(f"vmap_np.shape: {vmap_np.shape}")


m = vmap_np
x, y, z = np.where(m)

# Create a figure for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the voxels where the value is True
ax.scatter(x, y, z, c="blue", marker="o", s=10)  # s is the size of each point

# Set plot labels and titles
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
ax.set_title("3D Voxel Visualization")

# Show the plot
plt.show()
