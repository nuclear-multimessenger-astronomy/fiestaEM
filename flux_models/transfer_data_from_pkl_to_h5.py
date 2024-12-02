import numpy as np
import pickle
import h5py
import os
from prompt_toolkit import prompt


filein = "./afterglowpy/gaussian/afterglowpy_raw_data.pkl"
fileout = "./afterglowpy/gaussian/afterglowpy_raw_data.h5"

with open(filein, "rb") as file:
        data = pickle.load(file)
        jet_type = data["jet_type"]
        times = data["times"]
        nus = data["nus"]
        parameter_distributions = data["parameter_distributions"]
        parameter_names = data["parameter_names"]
        train_X_raw, train_y_raw = data["train_X_raw"], data["train_y_raw"]
        val_X_raw, val_y_raw = data["val_X_raw"], data["val_y_raw"]
        test_X_raw, test_y_raw = data["test_X_raw"], data["test_y_raw"]

print("Loaded data from file.")
if os.path.exists(fileout):
    user_input = prompt("Warning, will overwrite existing h5 file. Continue?")
    user_input = user_input.strip().lower()
    if user_input not in ["y", "yes"]:
        exit()

with h5py.File(fileout, "w") as f:
    f.create_dataset("times", data = times)
    f.create_dataset("nus", data = nus)
    f.create_dataset("parameter_names", data = parameter_names)
    f.create_dataset("parameter_distributions", data = str(parameter_distributions))
    f.create_dataset("jet_type", data = jet_type)
    f.create_group("train"); f.create_group("val"); f.create_group("test"); f.create_group("special_train")
    
    f["train"].create_dataset("X", data = train_X_raw, maxshape=(None, len(parameter_names)), chunks = (1000, len(parameter_names)))
    f["train"].create_dataset("y", data = train_y_raw, maxshape=(None, len(times)*len(nus)), chunks = (1000, len(times)*len(nus)))

    f["val"].create_dataset("X", data = val_X_raw, maxshape=(None, len(parameter_names)), chunks = (1000, len(parameter_names)))
    f["val"].create_dataset("y", data = val_y_raw, maxshape=(None, len(times)*len(nus)), chunks = (1000, len(times)*len(nus)))

    f["test"].create_dataset("X", data = test_X_raw, maxshape=(None, len(parameter_names)), chunks = (1000, len(parameter_names)))
    f["test"].create_dataset("y", data = test_y_raw, maxshape=(None, len(times)*len(nus)), chunks = (1000, len(times)*len(nus)))