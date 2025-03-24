import os
import h5py
import numpy as np
import re
import shutil
import tqdm

#####

directory = "./model"

#####

outfile = os.path.join(directory, "special_raw_data.h5")
pattern = r"_\d+\.h5$"  # Matches an underscore, digits, and ".h5" at the end
file_list = [f for f in os.listdir(directory) if bool(re.search(pattern, f))]
shutil.copy(os.path.join(directory, file_list[0]), outfile)

with h5py.File(outfile, "a") as f:
    
    for file in tqdm.tqdm(file_list[1:]):
        file = h5py.File(os.path.join(directory, file))
        for label in ["01"]:
            X = file["special_train"][label]["X"]
            Xset = f["special_train"][label]["X"]
            Xset.resize(Xset.shape[0]+X.shape[0], axis = 0)
            Xset[-X.shape[0]:] = X

            y = file["special_train"][label]["y"]
            yset = f["special_train"][label]["y"]
            yset.resize(yset.shape[0]+y.shape[0], axis = 0)
            yset[-y.shape[0]:] = y
        file.close()
    
    print("special train: ", f["special_train"]["01"]["y"].shape[0])


