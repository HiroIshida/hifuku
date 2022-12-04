import pickle

import matplotlib.pyplot as plt

with open("/tmp/hifuku_bench.pkl", "rb") as f:
    (naives, mines) = pickle.load(f)

fig, ax = plt.subplots()
ax.plot(naives, "o", label="naive")
ax.plot(mines, "x", label="proposed")
ax.legend()
ax.set_xlabel("each trial [-]")
ax.set_ylabel("computation time [sec]")
plt.show()
