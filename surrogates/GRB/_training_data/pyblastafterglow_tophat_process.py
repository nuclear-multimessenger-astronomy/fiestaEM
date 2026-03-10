import numpy as np
import h5py

with h5py.File("./pyblastafterglow_tophat_raw_data.h5", "r+") as f:
    for group in ["val", "test"]:
        f[group]["y"][:] = np.maximum(f[group]["y"][:], -50)

with h5py.File("./pyblastafterglow_tophat_raw_data.h5", "r+") as f:
    f["train"]["y"][:20_000] = np.maximum(f["train"]["y"][:20_000], -50)
    f["train"]["y"][20_000:40_000] = np.maximum(f["train"]["y"][20_000:40_000], -50)
    f["train"]["y"][40_000:] = np.maximum(f["train"]["y"][40_000:], -50)