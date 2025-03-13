"""Method to train the surrogate models"""

import dill
import os
import pickle
from typing import Callable, Dict

import numpy as np
import matplotlib.pyplot as plt

import jax
from jaxtyping import Array, Float, Int

from fiesta.utils import MinMaxScalerJax
from fiesta.filters import Filter
from fiesta.train.DataManager import DataManager
import fiesta.train.neuralnets as fiesta_nn

################
# TRAINING API #
################

class LightcurveTrainer:
    """Abstract class for training a collection of surrogate models per filter"""
   
    name: str
    outdir: str
    filters: list[Filter]
    parameter_names: list[str]

    train_X: Float[Array, "n_train"]
    train_y: Dict[str, Float[Array, "n"]]
    val_X: Float[Array, "n_val"]
    val_y: Dict[str, Float[Array, "n"]]
    
    def __init__(self, 
                 name: str,
                 outdir: str,
                 plots_dir: str = None,
                 save_preprocessed_data: bool = False) -> None:
        
        self.name = name
        # Check if directories exists, otherwise, create:
        self.outdir = outdir
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        self.plots_dir = plots_dir
        if self.plots_dir is not None and not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

        self.save_preprocessed_data = save_preprocessed_data

        # To be loaded by child classes
        self.filters = None
        self.parameter_names = None
        
        self.train_X = None
        self.train_y = None

        self.val_X = None
        self.val_y = None

    def __repr__(self) -> str:
        return f"LightcurveTrainer(name={self.name})"
    
    def preprocess(self):
        
        print("Preprocessing data by minmax scaling . . .")
        self.X_scaler = MinMaxScalerJax()
        self.X = self.X_scaler.fit_transform(self.X_raw)
        
        self.y_scaler: dict[str, MinMaxScalerJax] = {}
        self.y = {}
        for filt in self.filters:
            y_scaler = MinMaxScalerJax()
            self.y[filt.name] = y_scaler.fit_transform(self.y_raw[filt.name])
            self.y_scaler[filt.name] = y_scaler
        print("Preprocessing data . . . done")
    
    def fit(self,
            config: fiesta_nn.NeuralnetConfig,
            key: jax.random.PRNGKey = jax.random.PRNGKey(0),
            verbose: bool = True) -> None:
        """
        The config controls which architecture is built and therefore should not be specified here.
        
        Args:
            config (nn.NeuralnetConfig, optional): _description_. Defaults to None.
        """

        self.config = config
        self.models = {}
        input_ndim = len(self.parameter_names)

        for filt in self.filters:

            print(f"\n\n Training {filt.name}... \n\n")
            
            # Create neural network and initialize the state
            net = fiesta_nn.MLP(config = config, input_ndim = input_ndim, key = key)

            # Perform training loop
            state, train_losses, val_losses = net.train_loop(self.train_X, self.train_y[filt.name], self.val_X, self.val_y[filt.name], verbose=verbose)
            self.models[filt.name] = net
    
            # Plot and save the plot if so desired
            if self.plots_dir is not None:
                plt.figure(figsize=(10, 5))
                ls = "-o"
                ms = 3
                plt.plot([i+1 for i in range(len(train_losses))], train_losses, ls, markersize=ms, label="Train", color="red")
                plt.plot([i+1 for i in range(len(val_losses))], val_losses, ls, markersize=ms, label="Validation", color="blue")
                plt.legend()
                plt.xlabel("Epoch")
                plt.ylabel("MSE loss")
                plt.yscale('log')
                plt.title("Learning curves")
                plt.savefig(os.path.join(self.plots_dir, f"learning_curves_{filt.name}.png"))
                plt.close()
        
    def save(self):
        """
        Save the trained model and all the used metadata to the outdir.
        """
        # Save the metadata
        meta_filename = os.path.join(self.outdir, f"{self.name}_metadata.pkl")

        save = {}

        save["times"] = self.times
        save["parameter_names"] = self.parameter_names
        save["parameter_distributions"] = self.parameter_distributions
        save["X_scaler"] = self.X_scaler
        save["y_scaler"] = self.y_scaler

        save["model_type"] = "MLP"

        with open(meta_filename, "wb") as meta_file:
            dill.dump(save, meta_file)
        
        # Save the NN
        for filt in self.filters:
            model = self.models[filt.name]
            model.save_model(outfile = os.path.join(self.outdir, f"{self.name}_{filt.name}.pkl"))
                
    def _save_preprocessed_data(self) -> None:
        print("Saving preprocessed data . . .")
        np.savez(os.path.join(self.outdir, f"{self.name}_preprocessed_data.npz"), train_X=self.train_X, train_y = self.train_y, val_X = self.val_X, val_y = self.val_y)
        print("Saving preprocessed data . . . done")
    
class SVDTrainer(LightcurveTrainer):
    
    def __init__(self,
                 name: str,
                 outdir: str,
                 filters: list[str],
                 data_manager_args: dict,
                 svd_ncoeff: Int = 50,
                 conversion: Callable = lambda x: x,
                 plots_dir: str = None,
                 save_preprocessed_data: bool = False) -> None:
        """
        Initialize the surrogate model trainer that decomposes the training data into its SVD coefficients. The initialization also takes care of reading data and preprocessing it, but does not automatically fit the model. Users may want to inspect the data before fitting the model.
        
        Args:
            name (str): Name of the surrogate model. Will be used 
            outdir (str): Directory where the trained surrogate model is to be saved.
            filters (list[str]): List of the filters for which the surrogate has to be trained. These have to be either bandpasses from sncosmo or specifiy the frequency through endign with GHz or keV.
            data_manager_args (dict): data_manager_args (dict): Arguments for the DataManager class instance that will be used to read the data from the .h5 file in outdir and preprocess it.
            svd_ncoeff (int, optional) : Number of SVD coefficients to use in data reduction during training. Defaults to 50.
            conversion (str): references how to convert the parameters for the training. Defaults to None, in which case it's the identity.
            plots_dir (str, optional): Directory where the plots of the training process will be saved. Defaults to None, which means no plots will be generated.
            save_preprocessed_data (bool, optional): If True, the preprocessed data (reduced, rescaled) will be saved in the outdir. Defaults to False.
        """

        super().__init__(name = name,
                         outdir = outdir,
                         plots_dir = plots_dir,
                         save_preprocessed_data = save_preprocessed_data)
        
        self.svd_ncoeff = svd_ncoeff

        self.conversion = conversion
        
        self.data_manager = DataManager(**data_manager_args)
        self.data_manager.print_file_info()
        self.data_manager.pass_meta_data(self)
        self.load_filters(filters)

        self.preprocess()
        if self.save_preprocessed_data:
            self._save_preprocessed_data()
    
    def load_filters(self, filters):
        self.filters = []
        for filt in filters:
            Filt = Filter(filt)
            if Filt.nus[0] < self.nus[0] or Filt.nus[-1] > self.nus[-1]:
                raise ValueError(f"Filter {filt} exceeds the frequency range of the training data.")
            self.filters.append(Filt)
        
    def preprocess(self):
        """
        Preprocessing method to get the SVD coefficients of the training and validation data. This includes scaling the inputs and outputs, as well as performing SVD decomposition.
        """
        print(f"Decomposing training data to SVD coefficients.")
        self.train_X, self.train_y, self.val_X, self.val_y, self.X_scaler, self.y_scaler = self.data_manager.preprocess_svd(self.svd_ncoeff, self.filters, self.conversion)
        for key in self.train_y.keys():
            if np.any(np.isnan(self.train_y[key])) or np.any(np.isnan(self.val_y[key])):
                raise ValueError(f"Data preprocessing for {key} introduced nans. Check raw data for nans of infs or vanishing variance in a specific entry.")
        print(f"Preprocessing data . . . done")
