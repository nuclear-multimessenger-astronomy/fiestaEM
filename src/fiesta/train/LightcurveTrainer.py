"""Method to train the surrogate models"""

import os
import numpy as np

import jax
import jax.numpy as jnp 
from jaxtyping import Array, Float, Int
from typing import Dict
from fiesta.utils import MinMaxScalerJax
from fiesta import utils
from fiesta.utils import Filter
from fiesta import conversions
from fiesta.constants import days_to_seconds, c
from fiesta import models_utilities
from fiesta.train.DataManager import DataManager
import fiesta.train.neuralnets as fiesta_nn

import matplotlib.pyplot as plt
import pickle
from typing import Callable
import tqdm

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
        
        self.y_scalers: dict[str, MinMaxScalerJax] = {}
        self.y = {}
        for filt in self.filters:
            y_scaler = MinMaxScalerJax()
            self.y[filt.name] = y_scaler.fit_transform(self.y_raw[filt.name])
            self.y_scalers[filt.name] = y_scaler
            
        # Save the metadata
        self.preprocessing_metadata["X_scaler_min"] = self.X_scaler.min_val 
        self.preprocessing_metadata["X_scaler_max"] = self.X_scaler.max_val
        self.preprocessing_metadata["y_scaler_min"] = {filt.name: self.y_scalers[filt.name].min_val for filt in self.filters}
        self.preprocessing_metadata["y_scaler_max"] = {filt.name: self.y_scalers[filt.name].max_val for filt in self.filters}
        print("Preprocessing data . . . done")
    
    def fit(self,
            config: fiesta_nn.NeuralnetConfig = None,
            key: jax.random.PRNGKey = jax.random.PRNGKey(0),
            verbose: bool = True) -> None:
        """
        The config controls which architecture is built and therefore should not be specified here.
        
        Args:
            config (nn.NeuralnetConfig, optional): _description_. Defaults to None.
        """
        
        # Get default choices if no config is given
        if config is None:
            config = fiesta_nn.NeuralnetConfig()
        self.config = config
            
        trained_states = {}

        input_ndim = len(self.parameter_names)
        for filt in self.filters:

            print(f"\n\n Training {filt.name}... \n\n")
            
            # Create neural network and initialize the state
            net = fiesta_nn.MLP(layer_sizes=config.layer_sizes)
            key, subkey = jax.random.split(key)
            state = fiesta_nn.create_train_state(net, jnp.ones(input_ndim), subkey, config)
            
            # Perform training loop
            state, train_losses, val_losses = fiesta_nn.train_loop(state, config, self.train_X, self.train_y[filt.name], self.val_X, self.val_y[filt.name], verbose=verbose)

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

            trained_states[filt.name] = state
            
        self.trained_states = trained_states
        
    def save(self):
        """
        Save the trained model and all the used metadata to the outdir.
        """
        # Save the metadata
        meta_filename = os.path.join(self.outdir, f"{self.name}_metadata.pkl")

        save = {}

        save["times"] = self.times
        save["parameter_names"] = self.parameter_names

        with open(meta_filename, "wb") as meta_file:
            pickle.dump(save, meta_file)
        
        # Save the NN
        for filt in self.filters:
            model = self.trained_states[filt.name]
            model.save_model(outfile = os.path.join(self.outdir, f"{self.name}_{filt.name}.pkl"))
            save[filt.name] = self.preprocessing_metadata[filt.name]
        

        
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
                 svd_ncoeff: Int = 10,
                 plots_dir: str = None,
                 save_preprocessed_data: bool = False) -> None:
        """
        Initialize the surrogate model trainer that uses an SVD. The initialization also takes care of reading data and preprocessing it, but does not automatically fit the model. Users may want to inspect the data before fitting the model.
        
        Args:
            name (str): Name of the surrogate model. Will be used 
            lc_dir (list[str]): Directory where all the raw light curve files, to be read and processed into a surrogate model.
            outdir (str): Directory where the trained surrogate model has to be saved.
            filters (list[str], optional): List of all the filters used in the light curve files and for which surrogate has to be trained. If None, all the filters will be used. Defaults to None.
            svd_ncoeff: int : Number of SVD coefficients to use in data reduction during training. Defaults to 10.
            validation_fraction (Float, optional): Fraction of the data to be used for validation. Defaults to 0.2.
            tmin (Float, optional): Minimum time in days of the light curve, all data before is discarded. Defaults to 0.05.
            tmax (Float, optional): Maximum time in days of the light curve, all data after is discarded. Defaults to 14.0.
            dt (Float, optional): Time step in the light curve. Defaults to 0.1.
            plots_dir (str, optional): Directory where the plots of the training process will be saved. Defaults to None, which means no plots will be generated.
            save_raw_data (bool, optional): If True, the raw data will be saved in the outdir. Defaults to False.
            save_preprocessed_data: If True, the preprocessed data (reduced, rescaled) will be saved in the outdir. Defaults to False.
        """

        super().__init__(name = name,
                         outdir = outdir,
                         plots_dir = plots_dir,
                         save_preprocessed_data = save_preprocessed_data)
        
        self.svd_ncoeff = svd_ncoeff
        
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
        Preprocessing method to get the SVD coefficients of the MinMaxScaler. This includes scaling the inputs and outputs, performing SVD decomposition, and saving the necessary metadata for later use.
        """
        print(f"Decomposing training data to SVD coefficients.")
        self.train_X, self.train_y, self.val_X, self.val_y, self.X_scaler, self.y_scaler = self.data_manager.preprocess_svd(self.svd_ncoeff, self.filters)
        for key in self.train_y.keys():
            if np.any(np.isnan(self.train_y[key])) or np.any(np.isnan(self.val_y[key])):
                raise ValueError(f"Data preprocessing for {key} introduced nans. Check raw data for nans of infs or vanishing variance in a specific entry.")
        print(f"Preprocessing data . . . done")
        
class BullaSurrogateTrainer(SVDSurrogateTrainer):
    
    _times_grid: Float[Array, "n_times"]
    extract_parameters_function: Callable
    data_dir: str
    
    # Check if supported
    def __init__(self,
                 name: str,
                 outdir: str,
                 filters: list[str] = None,
                 data_dir: list[str] = None,
                 svd_ncoeff: Int = 10, 
                 validation_fraction: Float = 0.2,
                 tmin: Float = None,
                 tmax: Float = None,
                 dt: Float = None,
                 plots_dir: str = None,
                 save_raw_data: bool = False,
                 save_preprocessed_data: bool = False):
        
        # Check if this version of Bulla is supported
        supported_models = list(models_utilities.SUPPORTED_BULLA_MODELS)
        if name not in supported_models:
            raise ValueError(f"Bulla model version {name} is not supported yet. Supported models are: {supported_models}")
        
        # Get the function to extract parameters
        self.extract_parameters_function = models_utilities.EXTRACT_PARAMETERS_FUNCTIONS[name]
        self.data_dir=data_dir
        
        super().__init__(name=name, 
                         outdir=outdir, 
                         filters=filters, 
                         svd_ncoeff=svd_ncoeff, 
                         validation_fraction=validation_fraction, 
                         tmin=tmin, 
                         tmax=tmax, 
                         dt=dt, 
                         plots_dir=plots_dir, 
                         save_raw_data=save_raw_data,
                         save_preprocessed_data=save_preprocessed_data)
        
        
    def load_times(self):
        """
        Fetch the time grid from the Bulla .dat files or create from given input
        """
        self._times_grid = utils.get_times_bulla_file(self.lc_files[0])
        if self.tmin is None or self.tmax is None or self.dt is None:
            print("No time range given, using grid times")
            self.times = self._times_grid
            self.tmin = self.times[0]
            self.tmax = self.times[-1]
            self.dt = self.times[1] - self.times[0]
        else:
            self.times = np.arange(self.tmin, self.tmax + self.dt, self.dt)
        
    def load_parameter_names(self):
        self.parameter_names = models_utilities.BULLA_PARAMETER_NAMES[self.name]
        
    def load_filters(self, filters: list[str] = None):
        """
        If no filters are given, we will read the filters from the first Bulla lightcurve file and assume all files have the same filters

        Args:
            filters (list[str], optional): List of filters to be used in the training. Defaults to None.
        """
        filenames: list[str] = os.listdir(self.data_dir)
        self.lc_files = [os.path.join(self.data_dir, f) for f in filenames if f.endswith(".dat")]
        if filters is None:
            filters = utils.get_filters_bulla_file(self.lc_files[0], drop_times=True)
        self.filters = []
        
        # Create Filters objects for each filter
        for filter in filters:
            self.filters.append(Filter(filter))
        
    def _read_files(self) -> tuple[dict[str, Float[Array, " n_batch n_params"]], Float[Array, "n_batch n_times"]]:
        """
        Read the photometry files and interpolate the NaNs. 
        Output will be an array of shape (n_filters, n_batch, n_times)

        Args:
            lc_files (list[str]): List of all the raw light curve files, to be read and processed into a surrogate model.
            
        Returns:
            tuple[dict[str, Float[Array, " n_batch n_times"]], Float[Array, "n_batch n_params"]]: First return value is an array of all the parameter values extracted from the files. Second return value is a dictionary containing the filters and corresponding light curve data which has shape (n_batch, n_times).
        """
        
        # Fetch the result for each filter and add it to already existing dataset
        data = {filt: [] for filt in self.filters}
        for i, filename in enumerate(tqdm.tqdm(self.lc_files)):
            # Get a dictionary with keys being the filters and values being the light curve data
            lc_data = utils.read_single_bulla_file(filename)
            for filt in self.filters:
                # TODO: improve this cumbersome thing
                this_data = lc_data[filt.name]
                if i == 0:
                    data[filt.name] = this_data
                else:
                    data[filt.name] = np.vstack((data[filt.name], this_data))
                    
            # Fetch the parameter values of this file
            params = self.extract_parameters_function(filename)
            # TODO: improve this cumbersome thing
            if i == 0:
                parameter_values = params
            else:
                parameter_values = np.vstack((parameter_values, params))
                
        return parameter_values, data
    
    def load_raw_data(self):
        print("Reading data files and interpolating NaNs . . .")
        X_raw, y = self._read_files()
        y_raw = utils.interpolate_nans(y, self._times_grid, self.times)
        if self.save_raw_data:
            np.savez(os.path.join(self.outdir, "raw_data.npz"), X_raw=X_raw, times=self.times, times_grid=self._times_grid, **y_raw)
        
        # split here into training and validating data
        self.n_val_data = int(self.validation_fraction*len(X_raw))
        self.n_training_data = len(X_raw) - self.n_val_data
        mask = np.zeros(len(X_raw) ,dtype = bool)
        mask[np.random.choice(len(X_raw), self.n_val_data, replace = False)] = True

        self.train_X_raw, self.val_X_raw = X_raw[~mask], X_raw[mask]
        self.train_y_raw, self.val_y_raw = {}, {}

        print("self.filters")
        print(self.filters)

        for filt in self.filters:
            self.train_y_raw[filt.name] = y_raw[filt.name][~mask]
            self.val_y_raw[filt.name] = y_raw[filt.name][mask]
