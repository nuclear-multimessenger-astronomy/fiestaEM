"""Method to train the surrogate models"""

import os
import numpy as np
import itertools
import pandas as pd
import jax
import jax.numpy as jnp 
from jaxtyping import Array, Float, Int
from beartype import beartype as typechecker
import tqdm
from fiesta.utils import MinMaxScalerJax
from sklearn.model_selection import train_test_split
from collections import defaultdict

from fiesta import utils
from fiesta import conversions
from fiesta.constants import days_to_seconds, c
from fiesta import models_utilities
import fiesta.train.neuralnets as fiesta_nn
import matplotlib.pyplot as plt
import joblib
import afterglowpy as grb

class SurrogateTrainer:
    """Abstract class for training the surrogate models"""
    
    name: str # Name given to the model, e.g. Bu2022Ye
    parameter_names: list[str] # Names of the input parameters
    X: Float[Array, "n_batch ndim_input"] # input training data
    y: dict[str, Float[Array, "n_batch ndim_output"]] # output training data, organized per filter
    filters: list[str]
    
    def __init__(self, 
                 name: str) -> None:
        self.name = name
        self.parameter_names = []
        
    def __repr__(self) -> str:
        return f"SurrogateTrainer(name={self.name})"
    
    def preprocess(self):
        raise NotImplementedError
    
    def fit(self):
        raise NotImplementedError

class BullaSurrogateTrainer(SurrogateTrainer):
    
    X_raw: Float[Array, "n_batch n_params"]
    y_raw: dict[str, Float[Array, "n_batch n_times"]]
    
    X: Float[Array, "n_batch n_params"]
    y: dict[str, Float[Array, "n_batch n_svd_coeff"]]
    
    def __init__(self,
                 name: str,
                 lc_dir: list[str],
                 outdir: str,
                 filters: list[str] = None,
                 svd_ncoeff: Int = 10, 
                 validation_fraction: Float = 0.2,
                 tmin: Float = None,
                 tmax: Float = None,
                 dt: Float = None,
                 plots_dir: str = None,
                 save_raw_data: bool = False
                 ):
        
        """
        Initialize the surrogate model trainer. The initialization also takes care of reading data and preprocessing it, but does not automatically fit the model. Users may want to inspect the data before fitting the model.
        
        Note: currently, only models of Bulla type .dat files are supported
        
        Args:
            name (str): Name of the surrogate model. Will be used 
            lc_dir (list[str]): Directory where all the raw light curve files, to be read and processed into a surrogate model.
            outdir (str): Directory where the trained surrogate model has to be saved.
            filters (list[str], optional): List of all the filters used in the light curve files and for which surrogate has to be trained. If None, all the filters will be used. Defaults to None.
            validation_fraction (Float, optional): Fraction of the data to be used for validation. Defaults to 0.2.
            tmin (Float, optional): Minimum time of the light curve, all data before is discarded. Defaults to 0.05.
            tmax (Float, optional): Maximum time of the light curve, all data after is discarded. Defaults to 14.0.
            dt (Float, optional): Time step in the light curve. Defaults to 0.1.
            plots_dir (str, optional): Directory where the plots of the training process will be saved. Defaults to None, which means no plots will be generated.
            save_raw_data (bool, optional): If True, the raw data will be saved in the outdir. Defaults to False.
        """
        
        # Check if supported
        supported_models = list(models_utilities.SUPPORTED_BULLA_MODELS)
        if name not in supported_models:
            raise ValueError(f"Model {name} is not supported yet. Supported models are: {supported_models}")
        
        super().__init__(name)
        self.extract_parameters_function = models_utilities.EXTRACT_PARAMETERS_FUNCTIONS[name]
        self.lc_dir = lc_dir
        self.outdir = outdir
        self.svd_ncoeff = svd_ncoeff
        self.validation_fraction = validation_fraction
        
        # Check if directory exists, otherwise, create it:
        self.plots_dir = plots_dir
        if self.plots_dir is not None and not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
        
        self.lc_files = [os.path.join(lc_dir, f) for f in os.listdir(lc_dir) if f.endswith(".dat")]

        # If no filters are given, we will read the filters from the first file and assume all files have the same filters
        if filters is None:
            filters = utils.get_filters_bulla_file(self.lc_files[0], drop_times=True)
        self.filters = filters
        
        # Fetch the time grid and mask it to the desired time range
        _times_grid = utils.get_times_bulla_file(self.lc_files[0])
        
        # Create time grid for interpolation and output
        if tmin is None or tmax is None or dt is None:
            print("No time range given, using grid times")
            self.times = _times_grid
        else:
            self.times = np.arange(tmin, tmax + dt, dt)
        
        # Fetch parameter names
        self.parameter_names = models_utilities.BULLA_PARAMETER_NAMES[name]
        
        self.preprocessing_metadata = {"X_scaler_min": {}, 
                                       "X_scaler_max": {}, 
                                       "y_scaler_min": {},
                                       "y_scaler_max": {},
                                       "VA": {},
                                       "nsvd_coeff": self.svd_ncoeff,
                                       "times": self.times}
        
        print("Reading data files and interpolating NaNs . . .")
        self.X_raw, y = self.read_files()
        self.y_raw = utils.interpolate_nans(y, _times_grid, self.times)

        if save_raw_data:
            print("Saving raw data")
            np.savez(os.path.join(outdir, "raw_data.npz"), X_raw=self.X_raw, times=self.times, times_grid=_times_grid, **self.y_raw)
            print("Saving raw data done")
        
        print("Preprocessing data . . .")
        self.preprocess()
        
    def __repr__(self) -> str:
        return f"BullaSurrogateTrainer(name={self.name}, lc_dir={self.lc_dir}, outdir={self.outdir}, filters={self.filters})"
    
    #####################
    ### PREPROCESSING ###
    #####################
    
    def read_files(self) -> tuple[dict[str, Float[Array, " n_batch n_params"]], Float[Array, "n_batch n_times"]]:
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
                this_data = lc_data[filt]
                if i == 0:
                    data[filt] = this_data
                else:
                    data[filt] = np.vstack((data[filt], this_data))
                    
            # Fetch the parameter values of this file
            params = self.extract_parameters_function(filename)
            # TODO: improve this cumbersome thing
            if i == 0:
                parameter_values = params
            else:
                parameter_values = np.vstack((parameter_values, params))
                
        return parameter_values, data
                
    def preprocess(self) -> tuple[Float[Array, "n_batch n_params"], dict[str, Float[Array, "n_batch nsvd_coeff"]]]:
        """_summary_
        Preprocess the data. This includes scaling the inputs and outputs, performing SVD decomposition, and saving the necessary metadata for later use.
        """
        # Scale inputs
        X_scaler = MinMaxScalerJax()
        X = X_scaler.fit_transform(self.X_raw)
        
        self.preprocessing_metadata["X_scaler_min"] = X_scaler.min_val
        self.preprocessing_metadata["X_scaler_max"] = X_scaler.max_val
        
        # Scale outputs, do SVD and save into y
        y = {filt: [] for filt in self.filters}
        for filt in tqdm.tqdm(self.filters):
            
            data_scaler = MinMaxScalerJax()
            data = data_scaler.fit_transform(self.y_raw[filt])
            self.preprocessing_metadata["y_scaler_min"][filt] = data_scaler.min_val
            self.preprocessing_metadata["y_scaler_max"][filt] = data_scaler.max_val
            
            # Do SVD decomposition
            # TODO: make SVD decomposition optional so that people can also easily train on the full lightcurve if they want to
            UA, _, VA = np.linalg.svd(data, full_matrices=True)
            VA = VA.T

            n, n = UA.shape
            m, m = VA.shape

            # This is taken over from NMMA
            cAmat = np.zeros((self.svd_ncoeff, n))
            cAvar = np.zeros((self.svd_ncoeff, n))
            for i in range(n):
                ErrorLevel = 1.0
                cAmat[:, i] = np.dot(
                    data[i, :], VA[:, : self.svd_ncoeff]
                )
                errors = ErrorLevel * np.ones_like(data[i, :])
                cAvar[:, i] = np.diag(
                    np.dot(
                        VA[:, : self.svd_ncoeff].T,
                        np.dot(np.diag(np.power(errors, 2.0)), VA[:, : self.svd_ncoeff]),
                    )
                )
            self.preprocessing_metadata["VA"][filt] = VA
            
            # Transpose to get the shape (n_batch, n_svd_coeff)
            y[filt] = cAmat.T 
            
        self.X = X 
        self.y = y
        
        return X, y
    
    ###############
    ### FITTING ###
    ###############
    
    def fit(self,
            config: fiesta_nn.NeuralnetConfig = None,
            key: jax.random.PRNGKey = jax.random.PRNGKey(0),
            verbose: bool = True):
        """
        
        The config controls which architecture is built and therefore should not be specified here.
        
        Args:
            config (nn.NeuralnetConfig, optional): _description_. Defaults to None.
        """
        
        # Get default choices if no config is given
        if config is None:
            config = fiesta_nn.NeuralnetConfig()
        self.config = config
            
        trained_states = {filt: None for filt in self.filters}
        for filt in self.filters:
            # Fetch the output data of this filter, and perform train-validation split on it
            y = self.y[filt]
            
            # Finally, convert to jnp.arrays
            X = jnp.array(self.X)
            y = jnp.array(y)
            
            train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=self.validation_fraction)
            
            input_ndim = len(self.parameter_names)

            # Create neural network and initialize the state
            net = fiesta_nn.MLP(layer_sizes=config.layer_sizes)
            key, subkey = jax.random.split(key)
            state = fiesta_nn.create_train_state(net, jnp.ones(input_ndim), subkey, config)
            
            # Perform training loop
            state, train_losses, val_losses = fiesta_nn.train_loop(state, config, train_X, train_y, val_X, val_y, verbose=verbose)

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
                plt.savefig(os.path.join(self.plots_dir, f"learning_curves_{filt}.png"))
                plt.close()

            trained_states[filt] = state
            
        self.trained_states = trained_states
        
    def save(self):
        """
        Save the trained model and all the used metadata to the outdir.
        """
        
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
            
        for filt in self.filters:
            model = self.trained_states[filt]
            fiesta_nn.save_model(model, self.config, out_name=self.outdir + f"{filt}.pkl")
            
        # TODO: improve saving of the scalers: saving the objects is not the best way to do it and breaks pickle
        joblib.dump(self.preprocessing_metadata, self.outdir + f"{self.name}.joblib")
        
        
        
# TODO: perhaps rename to *_1D, since it is only for 1D light curves, and we might want to get support for 2D by incorporating the frequencies... Unsure about the approach here

# TODO: add support for providing filters, and take the nu range from there?
class AfterglowpyTrainer(SurrogateTrainer):
    
    times: Float[Array, "n_times"]
    X: Float[Array, "n_batch n_params"]
    y: dict[str, Float[Array, "n_batch n_times"]]
    
    def __init__(self,
                 name: str,
                 outdir: str,
                 prior_ranges: dict[str, list[Float, Float]],
                 filters: list[str],
                 n_grid: Int = 3,
                 jet_type: Int = -1,
                 tmin: Float = 0.1,
                 tmax: Float = 1000,
                 n_times: Int = 100,
                 use_log_spacing: bool = True,
                 validation_fraction: float = 0.2,
                 fixed_parameters: dict[str, Float] = {},
                 plots_dir: str = None,
                 save_data: bool = True,
                 load_data: bool = False,
                 ):
        """
        TODO: add documentation
        Initialize the surrogate model trainer. The initialization also takes care of reading data and preprocessing it, but does not automatically fit the model. Users may want to inspect the data before fitting the model.
        
        Args:
            name (str): Name given to the model
            outdir (str): Output directory to save the trained model
            prior_ranges (dict[str, list[Float, Float]]): Dictionary containing the prior ranges for each parameter, i.e., the range on which the surrogate must be trained. The keys should be the parameter names and the values should be a list containing the minimum and maximum values of the prior range. NOTE: frequency (nu) should be included, or given to fixed parameters.
        """
        
        super().__init__(name)
        
        self.plots_dir = plots_dir
        self.prior_ranges = prior_ranges
        self.parameter_names = list(prior_ranges.keys())
        self.filters = filters
        self.n_grid = n_grid
        self.save_data = save_data
        self.validation_fraction = validation_fraction
        self.fixed_parameters = fixed_parameters
        self.outdir = outdir
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
            
        # TODO: need to check if supported ids are correct?
        # Check jet type before saving
        supported_jet_types = [-1, 0, 1, 4]
        if jet_type not in supported_jet_types:
            raise ValueError(f"Jet type {jet_type} is not supported. Supported jet types are: {supported_jet_types}")
        self.jet_type = jet_type
        
        # Construct times
        self.tmin = tmin
        self.tmax = tmax
        self.n_times = n_times
        if use_log_spacing:
            times = np.logspace(np.log10(tmin), np.log10(tmax), num=n_times)
        else:
            times = np.linspace(tmin, tmax, num=n_times)
        self.times = times
        self._times_afterglowpy = self.times * days_to_seconds # afterglowpy takes seconds as input
        
        # Create the nus for the given filters
        filts, lambdas = utils.get_default_filts_lambdas(filters)
        self.filters = filts
        nus = c / lambdas
        self.nus = dict(zip(filts, nus))
        
        self.preprocessing_metadata = {"X_scaler_min": {}, 
                                       "X_scaler_max": {}, 
                                       "y_scaler_min": {},
                                       "y_scaler_max": {},
                                       "times": self.times,
                                       "nus": self.nus,
                                       "jet_type": self.jet_type,
                                       "parameter_names": self.parameter_names}
        
        # Load data
        if load_data:
            data = np.load(os.path.join(self.outdir, "data.npz"))
            self.X_raw = data["X_raw"]
            self.y_raw = {} 
            for filt in self.filters:
                if filt in data.keys():
                    self.y_raw[filt] = data[filt]
                else:
                    print(f"Filter {filt} is not in the data, removing from list")
                    self.filters.remove(filt)
        else:
            self.create_training_data()
        
        self.preprocess()
        
        if save_data and not load_data:
            np.savez(os.path.join(self.outdir, "data.npz"), X_raw=self.X_raw, times=self.times, nus=self.nus, **self.y_raw)
            
        
        
    def create_training_data(self):
        """
        Create a grid of training data with specified settings and generate the output files for them. 
        """

        # Generate a grid of parameters
        param_values = [np.linspace(v[0], v[1], self.n_grid) for v in self.prior_ranges.values()]

        # Generate a grid of all parameter combinations
        X_raw = list(itertools.product(*param_values))
        X_raw = [list(combination) for combination in X_raw]
        X_raw = np.array(X_raw)
        
        parameter_names = list(self.prior_ranges.keys())

        # TODO: for now we train per filter, but best to change this!
        # Create the output dataset
        print("Creating the output dataset . . .")
        y_raw = {}
        
        for filt in self.filters:
            output = np.empty((len(X_raw), len(self.times)))
            counter = 0
            nu = self.nus[filt]
            for param in tqdm.tqdm(X_raw):
                param_dict = dict(zip(parameter_names, param))
                
                # Update with parameters that afterglowpy needs and frequency for this filter
                param_dict.update(self.fixed_parameters)
                param_dict["nu"] = nu
                
                mJys = self.call_afterglowpy(param_dict)
                output[counter] = mJys
                counter += 1
            
            y_raw[filt] = output
        
        self.X_raw = X_raw
        self.y_raw = y_raw
        
    def preprocess(self):
    
        # Preprocessing: apply minmax-scaling
        self.X_scaler = MinMaxScalerJax()
        self.X = self.X_scaler.fit_transform(self.X_raw)
        
        self.y_scalers: dict[str, MinMaxScalerJax] = {}
        self.y = {}
        for filt in self.filters:
            y_scaler = MinMaxScalerJax()
            self.y[filt] = y_scaler.fit_transform(self.y_raw[filt])
            self.y_scalers[filt] = y_scaler
            
        # Save the metadata
        self.preprocessing_metadata["X_scaler_min"] = self.X_scaler.min_val 
        self.preprocessing_metadata["X_scaler_max"] = self.X_scaler.max_val
        self.preprocessing_metadata["y_scaler_min"] = {filt: self.y_scalers[filt].min_val for filt in self.filters}
        self.preprocessing_metadata["y_scaler_max"] = {filt: self.y_scalers[filt].max_val for filt in self.filters}
        
        
    def call_afterglowpy(self,
                         params_dict: dict[str, Float]) -> Float[Array, "n_times"]:
        """
        Call afterglowpy to generate a single flux density output, for a given set of parameters. Note that the parameters_dict should contain all the parameters that the model requires, as well as the nu value.
        The output will be a set of mJys.

        Args:
            Float[Array, "n_times"]: The flux density in mJys at the given times.
        """
        
        # TODO: move to lightcurvemodel?
        # # Create the time grid
        # t = np.empty((self.n_times, self.N_nu))
        # t[:, :] = np.logspace(np.log10(self.tmin), np.log10(self.tmax), num=self.n_times)[:, None]
        
        # # Create the nu grid
        # nu = np.empty((self.n_times, self.N_nu))
        # for idx in range(self.N_nu):
        #     nu[:, idx] = self.nus[self.filters[idx]]
        
        # Preprocess the params_dict into the format that afterglowpy expects, which is usually called Z
        Z = {}
        
        Z["jetType"]  = params_dict.get("jetType", self.jet_type)
        Z["specType"] = params_dict.get("specType", 0)
        Z["z"] = params_dict.get("z", 0.0)
        Z["xi_N"] = params_dict.get("xi_N", 1.0)
            
        Z["E0"]        = 10 ** params_dict["log10_E0"]
        Z["thetaCore"] = params_dict["thetaCore"]
        Z["n0"]        = 10 ** params_dict["log10_n0"]
        Z["p"]         = params_dict["p"]
        Z["epsilon_e"] = 10 ** params_dict["log10_epsilon_e"]
        Z["epsilon_B"] = 10 ** params_dict["log10_epsilon_B"]
        Z["d_L"]       = conversions.Mpc_to_cm(params_dict["luminosity_distance"])
        if "inclination_EM" in list(params_dict.keys()):
            Z["thetaObs"]  = params_dict["inclination_EM"]
        else:
            Z["thetaObs"]  = params_dict["thetaObs"]
        if self.jet_type == 1 or self.jet_type == 4:
            Z["b"] = params_dict["b"]
        if "thetaWing" in list(params_dict.keys()):
            Z["thetaWing"] = params_dict["thetaWing"]
        
        # Call afterglowpy, get the flux
        
        mJys = grb.fluxDensity(self._times_afterglowpy, params_dict["nu"], **Z)
        return mJys
    
    
    # TODO: there might be some code duplication here. Check how much is shared between this and Bulla and see if we can easily move it into the abstract class
    
    ###############
    ### FITTING ###
    ###############
    
    def fit(self,
            config: fiesta_nn.NeuralnetConfig = None,
            key: jax.random.PRNGKey = jax.random.PRNGKey(0),
            verbose: bool = True):
        """
        The config controls which architecture is built and therefore should not be specified here.
        
        Args:
            config (nn.NeuralnetConfig, optional): _description_. Defaults to None.
        """
        
        # Get default choices if no config is given
        if config is None:
            config = fiesta_nn.NeuralnetConfig()
        self.config = config
        
        print("config")
        print(config)
            
        trained_states = {filt: None for filt in self.filters}
        for filt in self.filters:
            # Fetch the output data of this filter, and perform train-validation split on it
            y = self.y[filt]
            
            # Finally, convert to jnp.arrays
            X = jnp.array(self.X)
            y = jnp.array(y)
            
            train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=self.validation_fraction)
            
            input_ndim = len(self.parameter_names)

            # Create neural network and initialize the state
            net = fiesta_nn.MLP(layer_sizes=config.layer_sizes)
            key, subkey = jax.random.split(key)
            state = fiesta_nn.create_train_state(net, jnp.ones(input_ndim), subkey, config)
            
            # Perform training loop
            state, train_losses, val_losses = fiesta_nn.train_loop(state, config, train_X, train_y, val_X, val_y, verbose=verbose)

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
                plt.savefig(os.path.join(self.plots_dir, f"learning_curves_{filt}.png"))
                plt.close()

            trained_states[filt] = state
            
        self.trained_states = trained_states
        
    def save(self):
        """
        Save the trained model and all the used metadata to the outdir.
        """
        
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
            
        for filt in self.filters:
            model = self.trained_states[filt]
            fiesta_nn.save_model(model, self.config, out_name=self.outdir + f"{filt}.pkl")
            
        # TODO: improve saving of the scalers: saving the objects is not the best way to do it and breaks pickle
        joblib.dump(self.preprocessing_metadata, self.outdir + f"{self.name}.joblib")