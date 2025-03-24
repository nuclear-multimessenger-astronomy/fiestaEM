import os
import h5py
import numpy as np
import shutil
import tqdm

#####

directory = "./model"

#####

outfile = os.path.join(directory, "pyblastafterglow_raw_data.h5")
file_list = [f for f in os.listdir(directory) if f.endswith(".h5")]
shutil.copy(os.path.join(directory, file_list[0]), outfile)

with h5py.File(outfile, "a") as f:
    
    for file in tqdm.tqdm(file_list[1:]):
        file = h5py.File(os.path.join(directory, file))
        for group in ["train", "val", "test"]:
            X = file[group]["X"]
            Xset = f[group]["X"]
            Xset.resize(Xset.shape[0]+X.shape[0], axis = 0)
            Xset[-X.shape[0]:] = X

            y = file[group]["y"]
            yset = f[group]["y"]
            yset.resize(yset.shape[0]+y.shape[0], axis = 0)
            yset[-y.shape[0]:] = y
        file.close()
    
    print("train: ", f["train"]["y"].shape[0])
    print("val: ", f["val"]["y"].shape[0])
    print("test: ", f["test"]["y"].shape[0])


