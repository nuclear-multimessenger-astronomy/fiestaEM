"""Method to train the surrogate models"""

import os
import numpy as np

import jax
import jax.numpy as jnp 
from jaxtyping import Array, Float, Int
from fiestaEM.utils import MinMaxScalerJax
from fiestaEM import utils
from fiestaEM.utils import Filter
from fiestaEM import conversions
from fiestaEM.constants import days_to_seconds, c
from fiestaEM import models_utilities
import fiestaEM.train.neuralnets as fiestaEM_nn

import matplotlib.pyplot as plt
import pickle
from typing import Callable
import tqdm

import afterglowpy as grb

class SurrogateTrainer:
    """Abstract class for training a collection of surrogate models per filter"""
    
    name: str
    outdir: str
    filters: list[Filter]
    parameter_names: list[str]
    
    validation_fraction: Float
    preprocessing_metadata: dict[str, dict[str, float]]
    
    # TODO: why do we have so many datasets?
    X: Float[Array, "n_batch n_input_surrogate"]
    y: dict[str, Float[Array, "n_batch n_output_surrogate"]]
    
    X_raw: Float[Array, "n_batch n_params"]
    y_raw: dict[str, Float[Array, "n_batch n_times"]]
    
    train_X: Float[Array, "n_batch n_params"]
    train_y: dict[str, Float[Array, "n_batch n_times"]]
    
    val_X: Float[Array, "n_batch n_params"]
    val_y: dict[str, Float[Array, "n_batch n_times"]]
    
    train_X_raw: Float[Array, "n_batch n_params"]
    train_y_raw: dict[str, Float[Array, "n_batch n_times"]]
    
    val_X_raw: Float[Array, "n_batch n_params"]
    val_y_raw: dict[str, Float[Array, "n_batch n_times"]]
    
    trained_states: dict[str, fiestaEM_nn.TrainState]
    
    def __init__(self, 
                 name: str,
                 outdir: str,
                 validation_fraction: Float = 0.2,
                 save_raw_data: bool = False,
                 save_preprocessed_data: bool = False) -> None:
        
        self.name = name
        self.outdir = outdir
        # Check if directories exists, otherwise, create:
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        
        self.save_raw_data = save_raw_data
        self.save_preprocessed_data = save_preprocessed_data
        
        # To be loaded by child classes
        self.filters = None
        self.parameter_names = None
        self.plots_dir = None
        
        self.validation_fraction = validation_fraction
        self.preprocessing_metadata = {}
        
        self.X_raw = None
        self.y_raw = None
        
        self.X = None
        self.y = None

    def __repr__(self) -> str:
        return f"SurrogateTrainer(name={self.name})"
    
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
            config: fiestaEM_nn.NeuralnetConfig = None,
            key: jax.random.PRNGKey = jax.random.PRNGKey(0),
            verbose: bool = True):
        """
        The config controls which architecture is built and therefore should not be specified here.
        
        Args:
            config (nn.NeuralnetConfig, optional): _description_. Defaults to None.
        """
        
        # Get default choices if no config is given
        if config is None:
            config = fiestaEM_nn.NeuralnetConfig()
        self.config = config
            
        trained_states = {}

        input_ndim = len(self.parameter_names)
        for filt in self.filters:
           
            # Create neural network and initialize the state
            net = fiestaEM_nn.MLP(layer_sizes=config.layer_sizes)
            key, subkey = jax.random.split(key)
            state = fiestaEM_nn.create_train_state(net, jnp.ones(input_ndim), subkey, config)
            
            # Perform training loop
            state, train_losses, val_losses = fiestaEM_nn.train_loop(state, config, self.train_X, self.train_y[filt.name], self.val_X, self.val_y[filt.name], verbose=verbose)

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
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        
        meta_filename = os.path.join(self.outdir, f"{self.name}_metadata.pkl")

        # FIXME: this should not be in this class
        if os.path.exists(meta_filename):
            with open(meta_filename, "rb") as meta_file:
                save = pickle.load(meta_file)
            if not np.array_equal(save["times"], self.times): # check whether the metadata from previously trained filters agrees
                raise Exception(f"The time array needs to coincide with the time array for previous filters: {save['times']}")
            if not np.array_equal(save["parameter_names"], self.parameter_names):
                 raise Exception(f"The parameters need to coincide with the parameters for previous filters: {save['parameter_names']}")
        else:
            save = {}

        save["times"] = self.times
        save["parameter_names"] = self.parameter_names
        # TODO: see if we can save the jet_type here somewhat more self-consistently

        for filt in self.filters:
            model = self.trained_states[filt.name]
            fiestaEM_nn.save_model(model, self.config, out_name=self.outdir + f"{filt.name}.pkl")
            save[filt.name] = self.preprocessing_metadata[filt.name]
        
        with open(meta_filename, "wb") as meta_file:
            pickle.dump(save, meta_file)
        
    def _save_raw_data(self):
        print("Saving raw data . . .")
        np.savez(os.path.join(self.outdir, "raw_data_training.npz"), X_raw=self.train_X_raw, **self.train_y_raw)
        np.savez(os.path.join(self.outdir, "raw_data_validation.npz"), X_raw=self.val_X_raw, **self.val_y_raw)
        print("Saving raw data . . . done")
        
    def _save_preprocessed_data(self):
        print("Saving preprocessed data . . .")
        np.savez(os.path.join(self.outdir, "preprocessed_data_training.npz"), X=self.train_X, **self.train_y)
        np.savez(os.path.join(self.outdir, "preprocessed_data_validation.npz"), X=self.val_X, **self.val_y)
        print("Saving preprocessed data . . . done")
    
class SVDSurrogateTrainer(SurrogateTrainer):
    
    outdir: str 
    svd_ncoeff: Int
    tmin: Float
    tmax: Float
    dt: Float
    times: Float[Array, "n_times"]
    plots_dir: str
    save_raw_data: bool
    save_preprocessed_data: bool
    
    def __init__(self,
                 name: str,
                 outdir: str,
                 filters: list[str] = None,
                 svd_ncoeff: Int = 10, 
                 validation_fraction: Float = 0.2,
                 tmin: Float = None,
                 tmax: Float = None,
                 dt: Float = None,
                 plots_dir: str = None,
                 save_raw_data: bool = False,
                 save_preprocessed_data: bool = False
                 ):
        
        """
        Initialize the surrogate model trainer that uses an SVD. The initialization also takes care of reading data and preprocessing it, but does not automatically fit the model. Users may want to inspect the data before fitting the model.
        
        Note: currently, only models of Bulla type .dat files are supported
        
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
        
        super().__init__(name=name, outdir=outdir, validation_fraction=validation_fraction)
        self.plots_dir = plots_dir
        
        if self.plots_dir is not None and not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
            
        self.svd_ncoeff = svd_ncoeff
        self.tmin = tmin
        self.tmax = tmax
        self.dt = dt
        self.plots_dir = plots_dir
        self.save_raw_data = save_raw_data
        self.save_preprocessed_data = save_preprocessed_data
        
        self.load_filters(filters)
        self.load_times()
        self.load_parameter_names()
            
        self.load_raw_data()

        self.preprocess()
    
        if save_preprocessed_data:
            self._save_preprocessed_data()
        if save_raw_data:
            self._save_raw_data()
        
    def load_parameter_names(self):
        raise NotImplementedError
        
    def load_times(self):
        raise NotImplementedError
    
    def load_filters(self, filters: list[str] = None):
        raise NotImplementedError
    
    def load_raw_data(self):
        raise NotImplementedError
        
    def preprocess(self):
        """
        Preprocess the data. This includes scaling the inputs and outputs, performing SVD decomposition, and saving the necessary metadata for later use.
        """
        # Scale inputs
        X_scaler = MinMaxScalerJax()
        self.train_X = X_scaler.fit_transform(self.train_X_raw) # fit the scaler to the training data
        self.val_X = X_scaler.transform(self.val_X_raw) # transform the val data
      
        # Scale outputs, do SVD and save into y
        self.train_y = {filt.name: [] for filt in self.filters}
        self.val_y = {filt.name: [] for filt in self.filters}
        
        print(f"Rescaling the training and validation data for filters {[filter.name for filter in self.filters]}")
        for filt in tqdm.tqdm(self.filters):
            
            y_scaler = MinMaxScalerJax()
            data = y_scaler.fit_transform(self.train_y_raw[filt.name])
            
            # Do SVD decomposition on the training data
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

            self.train_y[filt.name] = cAmat.T # Transpose to get the shape (n_batch, n_svd_coeff)

            # Do SVD decomposition on the validation data
            val_data = y_scaler.transform(self.val_y_raw[filt.name])
            cAmat = np.zeros((self.svd_ncoeff, self.n_val_data))
            for i in range(self.n_val_data):
                cAmat[:,i] = np.dot(
                    val_data[i,:], VA[:, : self.svd_ncoeff]
                )
            
            self.val_y[filt.name] = cAmat.T # Transpose to get the shape (n_val, n_svd_coeff)

            #Save the scalers
            self.preprocessing_metadata[filt.name] = {"VA": VA, "X_scaler_max": X_scaler.max_val, "X_scaler_min": X_scaler.min_val, "y_scaler": y_scaler, "svd_ncoeff": self.svd_ncoeff}

        
    def __repr__(self) -> str:
        return f"SVDSurrogateTrainer(name={self.name}, lc_dir={self.lc_dir}, outdir={self.outdir}, filters={self.filters})"
    
        
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

        
# TODO: perhaps rename to *_1D, since it is only for 1D light curves, and we might want to get support for 2D by incorporating the frequencies... Unsure about the approach here

class AfterglowpyTrainer(SVDSurrogateTrainer):
    
    parameter_grid: dict[str, list[Float]]
    n_training_data: Int
    fixed_parameters: dict[str, Float]
    jet_type: Int
    use_log_spacing: bool
    
    _times_afterglowpy: Float[Array, "n_times"]
    nus: dict[str, Float]
    
    def __init__(self,
                 name: str,
                 outdir: str,
                 filters: list[str],
                 parameter_grid: dict[str, list[float]],
                 n_training_data: Int = 5000,
                 jet_type: Int = -1,
                 fixed_parameters: dict[str, Float] = {},
                 tmin: Float = 0.1,
                 tmax: Float = 1000,
                 n_times: Int = 100,
                 use_log_spacing: bool = True,
                 validation_fraction: float = 0.2,
                 plots_dir: str = None,
                 svd_ncoeff: Int = 10,
                 save_raw_data: bool = False,
                 save_preprocessed_data: bool = False,
                 remake_training_data = False,
                 ):
        """
        Initialize the surrogate model trainer. The initialization also takes care of reading data and preprocessing it, but does not automatically fit the model. Users may want to inspect the data before fitting the model.
        
        Args:
            name (str): Name given to the model
            outdir (str): Output directory to save the trained model
            parameter_grid (dict[str, list[Float]]): Dictionary containing the grid points for each parameter, i.e., the parameter values on which the surrogate will be trained. The keys should be the parameter names and the values should be a list..
            jet_type (Int): Type of jet for the afterglowpy, -1 is tophat, 0 is Gaussian, 4 is PowerLaw
            fixed_parameters (dict[str, Float]) : values of the afterglowpy parameters that should be kept fixed for the surrogate model
            tmin (Float, optional): Minimum time in days of the light curve, all data before is discarded. Defaults to 0.1.
            tmax (Float, optional): Maximum time in days of the light curve, all data after is discarded. Defaults to 1000.
            n_times: number of time nodes for the training light curve data
            use_log_spacing: bool : whether the time nodes of the training light curve data should be log10 spaced
            validation_fraction (Float, optional): Fraction of the data to be used for validation. Defaults to 0.2.
            plots_dir : str : outdir for the plots
            svd_ncoeff: int : Number of SVD coefficients to use in data reduction during training. Defaults to 10.
            save_raw_data (bool, optional): If True, the raw data will be saved in the outdir. Defaults to False.
            save_preprocessed_data: If True, the preprocessed data (reduced, rescaled) will be saved in the outdir. Defaults to False.
        """
        

        self.n_times = n_times
        dt = (tmax - tmin) / n_times
        self.parameter_grid = parameter_grid
        self.fixed_parameters = fixed_parameters
        self.use_log_spacing = use_log_spacing
        
        # Check jet type before saving
        supported_jet_types = [-1, 0, 4]
        if jet_type not in supported_jet_types:
            raise ValueError(f"Jet type {jet_type} is not supported. Supported jet types are: {supported_jet_types}")
        self.jet_type = jet_type
        self.remake_training_data = remake_training_data

        self.n_training_data = n_training_data
        self.validation_fraction = validation_fraction
        self.n_val_data = int(self.n_training_data * self.validation_fraction/(1-self.validation_fraction))
            
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
        
    def load_filters(self, filters: list[str]):
        self.filters = []
        for filter in filters:
            try:
                self.filters.append(Filter(filter))
            except:
                raise Exception(f"Filter {filter} not available.")

    def load_times(self):
        if self.use_log_spacing:
            times = np.logspace(np.log10(self.tmin), np.log10(self.tmax), num=self.n_times)
        else:
            times = np.linspace(self.tmin, self.tmax, num=self.n_times)
        self.times = times
        self._times_afterglowpy = self.times * days_to_seconds # afterglowpy takes seconds as input
        
    def load_parameter_names(self):
        self.parameter_names = list(self.parameter_grid.keys())
    
    def load_raw_data(self):
        data_files_exist =  os.path.exists(self.outdir+"/raw_data_training.npz") and os.path.exists(self.outdir+"/raw_data_validation.npz")
        if data_files_exist and not self.remake_training_data:
            self.train_X_raw, self.train_y_raw, self.val_X_raw, self.val_y_raw = self._read_files()
        else:
            self.create_raw_data()
    
    def create_raw_data(self):
        """
        Create a grid of training data with specified settings and generate the output files for them.

        TODO: for now we train per filter, but best to change this!
        """
        # Create training data
        X_raw = np.empty((self.n_training_data, len(self.parameter_names)))
        y_raw = {filt.name: np.empty((self.n_training_data, len(self.times))) for filt in self.filters}

        for j, key in enumerate(self.parameter_grid.keys()):
            X_raw[:,j] = np.random.choice(self.parameter_grid[key], size = self.n_training_data, replace = True)


        print(f"Creating the afterglowpy training dataset on grid with {self.n_training_data} points.")
        for i in tqdm.tqdm(range(self.n_training_data)):
            for filt in self.filters:
                param_dict = dict(zip(self.parameter_names, X_raw[i]))
                param_dict.update(self.fixed_parameters)
                param_dict["nu"] = filt.nu # Add nu per filter before calling afterglowpy
                
                # Create and save output
                mJys = self._call_afterglowpy(param_dict)
                y_raw[filt.name][i] = conversions.mJys_to_mag_np(mJys)
                
        self.train_X_raw = X_raw
        self.train_y_raw = y_raw


        # Create validation data
        X_raw = np.empty((self.n_val_data, len(self.parameter_names)))
        y_raw = {filt.name: np.empty((self.n_val_data, len(self.times))) for filt in self.filters}

        print(f"Creating the afterglowpy validation dataset on {self.n_val_data} random points within grid.")
        for i in tqdm.tqdm(range(self.n_val_data)):
            X_raw[i] = [np.random.uniform(self.parameter_grid[p][0], self.parameter_grid[p][-1]) for p in self.parameter_names]          

            for filt in self.filters:
                param_dict = dict(zip(self.parameter_names, X_raw[i]))
                param_dict.update(self.fixed_parameters)
                param_dict["nu"] = filt.nu # Add nu per filter before calling afterglowpy
                
                # Create and save output
                mJys = self._call_afterglowpy(param_dict)
                y_raw[filt.name][i] = conversions.mJys_to_mag_np(mJys)

        self.val_X_raw = X_raw
        self.val_y_raw = y_raw
       


    def _read_files(self,):
        raw_data_train = np.load(self.outdir+"/raw_data_training.npz")
        raw_data_validation = np.load(self.outdir+'/raw_data_validation.npz')

        self.n_training_data = 4000
        self.n_val_data = 1000

        training_y_raw = {}
        val_y_raw = {}
        
        select = np.random.choice(range(0, len(raw_data_train["X_raw"])), size = self.n_training_data, replace = False)
        select_val = np.random.choice(range(0, len(raw_data_validation["X_raw"])), size = self.n_val_data, replace = False)
        
        for filt in self.filters:
            training_y_raw[filt.name] = raw_data_train[filt.name][select]
            val_y_raw[filt.name] = raw_data_validation[filt.name][select_val]
        return raw_data_train["X_raw"][select], training_y_raw, raw_data_validation["X_raw"][select_val], val_y_raw
        
        
    def _call_afterglowpy(self,
                         params_dict: dict[str, Float]) -> Float[Array, "n_times"]:
        """
        Call afterglowpy to generate a single flux density output, for a given set of parameters. Note that the parameters_dict should contain all the parameters that the model requires, as well as the nu value.
        The output will be a set of mJys.

        Args:
            Float[Array, "n_times"]: The flux density in mJys at the given times.
        """
        
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
        Z["d_L"]       = 3.086e19 # fix at 10 pc, so that AB magnitude equals absolute magnitude
        if "inclination_EM" in list(params_dict.keys()):
            Z["thetaObs"]  = params_dict["inclination_EM"]
        else:
            Z["thetaObs"]  = params_dict["thetaObs"]
        if self.jet_type == 1 or self.jet_type == 4:
            Z["b"] = params_dict["b"]
        if "thetaWing" in list(params_dict.keys()):
            Z["thetaWing"] = params_dict["thetaWing"]
        
        # Afterglowpy returns flux in mJys
        mJys = grb.fluxDensity(self._times_afterglowpy, params_dict["nu"], **Z)
        return mJys