import numpy as np
import corner
import matplotlib.pyplot as plt
import h5py

default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16),
                        #color = "blue"
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=False, 
                        plot_datapoints=True, 
                        plot_contours = False,
                        fill_contours=False,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        save=False,
                        truth_color="red")

#######

file = "./model/pyblastafterglow_raw_data.h5"

#######


with h5py.File(file, "r") as f:
    X = f["train"]["X"][:]
    parameter_names = f["parameter_names"][()]


corner.corner(X, **default_corner_kwargs, labels = parameter_names)
plt.show()
breakpoint()