"""Method to train the surrogate models"""

import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp 
from jaxtyping import Array, Float, Int
from beartype import beartype as typechecker
import tqdm
from fiesta.utils import MinMaxScalerJax
from sklearn.model_selection import train_test_split
from collections import defaultdict

from fiesta.train import utils
import fiesta.train.neuralnets as fiesta_nn
import matplotlib.pyplot as plt
import joblib

class SurrogateTrainer:
    """Abstract class for training the surrogate models"""
    
    # TODO: change n_files to n_batch?
    name: str # Name given to the model, e.g. Bu2022Ye
    parameter_names: list[str] # Names of the input parameters
    X: Float[Array, "n_batch ndim_input"] # input training data
    y: dict[str, Float[Array, "n_batch ndim_output"]] # output training data, organized per filter
    
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
    
    X_raw: Float[Array, "n_files n_params"]
    y_raw: dict[str, Float[Array, "n_files n_times"]]
    
    X: Float[Array, "n_files n_params"]
    y: dict[str, Float[Array, "n_files n_svd_coeff"]]
    
    def __init__(self,
                 name: str,
                 lc_dir: list[str],
                 outdir: str,
                 filters: list[str] = None,
                 svd_ncoeff: Int = 10, 
                 validation_fraction: Float = 0.2,
                 plots_dir: str = None
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
            plots_dir (str, optional): Directory where the plots of the training process will be saved. Defaults to None, which means no plots will be generated.
        """
        
        # Check if supported
        supported_models = list(utils.BULLA_PARAMETER_NAMES.keys())
        if name not in supported_models:
            raise ValueError(f"Model {name} is not supported yet. Supported models are: {supported_models}")
        
        super().__init__(name)
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
        
        # Fetch times
        self.times = utils.get_times_bulla_file(self.lc_files[0])
        
        # Assert all times are same
        for filename in self.lc_files:
            assert np.allclose(self.times, utils.get_times_bulla_file(filename)), "All the times should be same"
            
        # Fetch parameter names
        self.parameter_names = utils.BULLA_PARAMETER_NAMES[name]
        
        # TODO: check if these have to be saved or not
        # self.preprocessing_metadata = {"X_scaler": {}, "y_scaler": {}, "cAmat": {}, "cAstd": {}, "VA": {}}
        self.preprocessing_metadata = {"X_scaler_min": {}, 
                                       "X_scaler_max": {}, 
                                       "y_scaler_min": {},
                                       "y_scaler_max": {},
                                       "VA": {}, "nsvd_coeff": self.svd_ncoeff}
        
        print("Reading data files and interpolating NaNs . . .")
        self.X_raw, self.y_raw = self.read_files()
        self.y_raw = utils.interpolate_nans(self.y_raw, self.times)
        
        print("Preprocessing data . . .")
        self.preprocess()
        
    def __repr__(self) -> str:
        return f"BullaSurrogateTrainer(name={self.name}, lc_dir={self.lc_dir}, outdir={self.outdir}, filters={self.filters})"
    
    #####################
    ### PREPROCESSING ###
    #####################
    
    def read_files(self) -> tuple[dict[str, Float[Array, " n_files n_params"]], Float[Array, "n_files n_times"]]:
        """
        Read the photometry files and interpolate the NaNs. 
        Output will be an array of shape (n_filters, n_files, n_times)

        Args:
            lc_files (list[str]): List of all the raw light curve files, to be read and processed into a surrogate model.
            
        Returns:
            tuple[dict[str, Float[Array, " n_files n_times"]], Float[Array, "n_files n_params"]]: First return value is an array of all the parameter values extracted from the files. Second return value is a dictionary containing the filters and corresponding light curve data which has shape (n_files, n_times).
        """
        
        # TODO: figure out how to save?
        data = {filt: [] for filt in self.filters}

        # Fetch the result for each filter and add it to already existing dataset
        for i, filename in enumerate(tqdm.tqdm(self.lc_files)):
            # Get a dictionary with keys being the filters and values being the light curve data
            lc_data = utils.read_single_bulla_file(filename)
            for filt in self.filters:
                if i == 0:
                    data[filt] = lc_data[filt]
                else:
                    data[filt] = np.vstack((data[filt], lc_data[filt]))
                    
            # Fetch the parameter values of this file
            # TODO: make this more general than Bu2022Ye once I figured out how to do it
            params = utils.extract_Bu2022Ye_parameters(filename)
            if i == 0:
                parameter_values = params
            else:
                parameter_values = np.vstack((parameter_values, params))
                
        return parameter_values, data
                
    def preprocess(self) -> tuple[Float[Array, "n_files n_params"], dict[str, Float[Array, "n_files nsvd_coeff"]]]:
        """_summary_
        Preprocess the data. This includes scaling the inputs and outputs, performing SVD decomposition, and saving the necessary metadata for later use.
        """
        # Scale inputs
        X_scaler = MinMaxScalerJax()
        X = X_scaler.fit_transform(self.X_raw)
        
        # Scale outputs, do SVD and save into y
        y = {filt: [] for filt in self.filters}
        for filt in tqdm.tqdm(self.filters):
            # TODO: this is now duplicate in each filter. Remove this, not used in the LC generators
            self.preprocessing_metadata["X_scaler_min"][filt] = X_scaler.min_val
            self.preprocessing_metadata["X_scaler_max"][filt] = X_scaler.max_val
            
            data_scaler = MinMaxScalerJax()
            data = data_scaler.fit_transform(self.y_raw[filt])
            self.preprocessing_metadata["y_scaler_min"][filt] = data_scaler.min_val
            self.preprocessing_metadata["y_scaler_max"][filt] = data_scaler.max_val
            
            # Do SVD decomposition
            # TODO: generalize this so that people can also easily train on the full lightcurve if they want to
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
            cAstd = np.sqrt(cAvar)
            
            self.preprocessing_metadata["VA"][filt] = VA
            # TODO: check if these have to be saved or not
            # self.preprocessing_metadata["cAmat"][filt] = cAmat
            # self.preprocessing_metadata["cAstd"][filt] = cAstd
            
            y[filt] = cAmat.T # Transpose to get the shape (n_files, n_svd_coeff)
            
        self.X = X 
        self.y = y
        
        return X, y
    
    ###############
    ### FITTING ###
    ###############
    
    def fit(self,
            config: fiesta_nn.NeuralnetConfig = None,
            key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
        """
        
        The config controls which architecture is built and therefore should not be specified here.
        # TODO: - make architecture also part of config, if changed later on?
        
        Args:
            config (nn.NeuralnetConfig, optional): _description_. Defaults to None.
        """
        
        # Get default choices if not given
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
            
            # TODO: do we want to fix the random seed?
            train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=self.validation_fraction) # random_state=42
            
            input_ndim = len(self.parameter_names)

            # Create neural network and initialize the state
            net = fiesta_nn.MLP(layer_sizes=config.layer_sizes)
            key, subkey = jax.random.split(key)
            state = fiesta_nn.create_train_state(net, jnp.ones(input_ndim), subkey, config)
            
            # Perform training loop
            state, train_losses, val_losses = fiesta_nn.train_loop(state, config, train_X, train_y, val_X, val_y)

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
                plt.show()

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
        
        
### TODO: move to another space
# def load_model_all_filts(svd_model: SVDTrainingModel, model_dir: str):
#     # Iterate over all the filters that are present in the SVD model
#     filters = list(svd_model.keys())
#     for filt in filters:
#         # Check whether we have a saved model for this filter
#         # TODO what if file extension changes?
#         filenames = [file for file in os.listdir(model_dir) if f"{filt}.pkl" in file]
#         if len(filenames) == 0:
#             raise ValueError(f"Error loading flax model: filter {filt} does not seem to be saved in directory {model_dir}")
#         elif len(filenames) > 1:
#             print(f"Warning: there are several matches with filter {filt} in directory {model_dir}, loading first")
#         # If we have a saved model, load in and save into our object
#         filename = filenames[0]
#         state = load_model(model_dir + filename)
#         svd_model[filt]["model"] = state