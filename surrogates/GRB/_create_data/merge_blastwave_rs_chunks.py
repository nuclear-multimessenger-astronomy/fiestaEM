"""Merge chunk HDF5 files into a single training data file with train/val/test splits.

Usage:
    python merge_blastwave_rs_chunks.py [n_val] [n_test]

Defaults: n_val=2000, n_test=2000. Everything else goes to train.
"""
import sys
import glob
import numpy as np
import h5py

DATADIR = "/fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/_training_data"

n_val = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

if n_val < 0 or n_test < 0:
    raise ValueError(f"Split sizes must be non-negative (got n_val={n_val}, n_test={n_test})")

chunk_pattern = f"{DATADIR}/blastwave_rs_gaussian_chunk_*.h5"
outfile = f"{DATADIR}/blastwave_rs_gaussian_raw_data.h5"

chunk_files = sorted(glob.glob(chunk_pattern))
print(f"Found {len(chunk_files)} chunk files")

if not chunk_files:
    raise RuntimeError(f"No chunk files found matching {chunk_pattern}")

# Read all chunks
all_X = []
all_y = []
skipped = []
for f in chunk_files:
    with h5py.File(f, "r") as hf:
        if "train" in hf and "X" in hf["train"]:
            all_X.append(hf["train"]["X"][:])
            all_y.append(hf["train"]["y"][:])
            print(f"  {f}: {hf['train']['X'].shape[0]} samples")
        else:
            skipped.append(f)
            print(f"  WARNING: {f} has no train/X group, skipping")

if skipped:
    print(f"\nWARNING: {len(skipped)} of {len(chunk_files)} chunks were skipped")

if not all_X:
    raise RuntimeError("No valid samples found in any chunk file")

X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)
print(f"\nTotal samples: {len(X)}")

# Shuffle
rng = np.random.default_rng(seed=123)
idx = rng.permutation(len(X))
X = X[idx]
y = y[idx]

# Split
n_total = len(X)
if n_total <= n_val + n_test:
    raise ValueError(f"Not enough samples ({n_total}) for val ({n_val}) + test ({n_test}) + training")

X_test, y_test = X[:n_test], y[:n_test]
X_val, y_val = X[n_test:n_test+n_val], y[n_test:n_test+n_val]
X_train, y_train = X[n_test+n_val:], y[n_test+n_val:]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Read metadata from first chunk
with h5py.File(chunk_files[0], "r") as hf:
    times = hf["times"][:]
    nus = hf["nus"][:]
    parameter_names = hf["parameter_names"][:].astype(str).tolist()
    parameter_distributions = hf["parameter_distributions"][()]
    jet_type = hf["jet_type"][()]

# Write merged file
chunk_size = 100
with h5py.File(outfile, "w") as hf:
    hf.create_dataset("times", data=times)
    hf.create_dataset("nus", data=nus)
    hf.create_dataset("parameter_names", data=parameter_names)
    hf.create_dataset("parameter_distributions", data=parameter_distributions)
    hf.create_dataset("jet_type", data=jet_type)

    for group_name, gX, gy in [("train", X_train, y_train),
                                ("val", X_val, y_val),
                                ("test", X_test, y_test)]:
        g = hf.create_group(group_name)
        n_params = X.shape[1]
        n_nus = y.shape[1]
        n_times = y.shape[2]
        if len(gX) > 0:
            g.create_dataset("X", data=gX,
                             maxshape=(None, n_params),
                             chunks=(min(chunk_size, len(gX)), n_params))
            g.create_dataset("y", data=gy,
                             maxshape=(None, n_nus, n_times),
                             chunks=(min(chunk_size, len(gy)), n_nus, n_times))
        else:
            g.create_dataset("X", shape=(0, n_params),
                             maxshape=(None, n_params), dtype="float64")
            g.create_dataset("y", shape=(0, n_nus, n_times),
                             maxshape=(None, n_nus, n_times), dtype="float64")

    hf.create_group("special_train")

print(f"\nMerged file written to {outfile}")

# Quick sanity check
with h5py.File(outfile, "r") as hf:
    for grp in ["train", "val", "test"]:
        if "X" in hf[grp]:
            print(f"  {grp}: X={hf[grp]['X'].shape}, y={hf[grp]['y'].shape}")
    yt = hf["train"]["y"][:]
    print(f"  y stats: min={np.nanmin(yt):.3f}, max={np.nanmax(yt):.3f}, "
          f"NaN={np.any(np.isnan(yt))}, inf={np.any(np.isinf(yt))}")
