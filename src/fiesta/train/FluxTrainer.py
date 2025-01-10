"""Method to train the surrogate models"""

import os
import numpy as np
import gc

import jax
import jax.numpy as jnp 
from jaxtyping import Array, Float, Int

from fiesta.utils import MinMaxScalerJax, StandardScalerJax, PCAdecomposer, ImageScaler
import fiesta.train.neuralnets as fiesta_nn

import matplotlib.pyplot as plt
import pickle
import h5py

################
# TRAINING API #
################

class FluxTrainer:
    """Abstract class for training a surrogate model that predicts a spectral flux density array."""
    
    name: str
    outdir: str
    parameter_names: list[str]

    train_X: Float[Array, "n_train"]
    train_y: Float[Array, "n_train"]
    val_X: Float[Array, "n_val"]
    val_y: Float[Array, "n_val"]
    
    def __init__(self, 
                 name: str,
                 outdir: str,
                 plots_dir: str = None,
                 save_preprocessed_data: bool = False
                 ) -> None:
        
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
        self.parameter_names = None
        
        self.train_X = None
        self.train_y = None

        self.val_X = None
        self.val_y = None

    def __repr__(self) -> str:
        return f"FluxTrainer(name={self.name})"
    
    def preprocess(self):
        raise NotImplementedError
    
    def fit(self, 
            config: fiesta_nn.NeuralnetConfig = None,
            key: jax.random.PRNGKey = jax.random.PRNGKey(0),
            verbose: bool = True) -> None:
        raise NotImplementedError
    
    def plot_learning_curve(self, train_losses, val_losses):
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
        plt.savefig(os.path.join(self.plots_dir, f"learning_curves_{self.name}.png"))
        plt.close()
    
    def save(self) -> None:
        """
        Save the trained model and all the metadata to the outdir.
        The meta data is saved as a pickled dict to be read by fiesta.inference.lightcurve_model.SurrogateLightcurveModel.
        The NN is saved as a pickled serialized dict using the NN.save_model method.
        """
        # Save the metadata
        meta_filename = os.path.join(self.outdir, f"{self.name}_metadata.pkl")
        
        save = {}
        save["times"] = self.times
        save["nus"] = self.nus
        save["parameter_names"] = self.parameter_names
        save["parameter_distributions"] = self.parameter_distributions
        save["X_scaler"] = self.X_scaler
        save["y_scaler"] = self.y_scaler

        with open(meta_filename, "wb") as meta_file:
            pickle.dump(save, meta_file)
        
        # Save the NN
        self.network.save_model(outfile = os.path.join(self.outdir, f"{self.name}.pkl"))
    
    def _save_preprocessed_data(self) -> None:
        print("Saving preprocessed data . . .")
        np.savez(os.path.join(self.outdir, f"{self.name}_preprocessed_data.npz"), train_X=self.train_X, train_y= self.train_y, val_X = self.val_X, val_y = self.val_y)
        print("Saving preprocessed data . . . done")

class PCATrainer(FluxTrainer):
    
    def __init__(self,
                 name: str,
                 outdir: str,
                 data_manager_args: dict,
                 n_pca: Int = 100,
                 plots_dir: str = None,
                 save_preprocessed_data: bool = False) -> None:
        """
        FluxTrainer for training a feed-forward neural network on the PCA coefficients of the training data to predict the full 2D spectral flux density array.
        Initializing will read the data and preprocess it with the DataManager class. It can then be fit with the fit() method. 
        To write the surrogate model to file, the save() method is to be used, which will create two pickle files (one for the metadata, one for the neural network).

        Args:
            name (str): Name of the model to be trained. Will be used when saving metadata and model to file.
            outdir (str): Directory where the NN and its metadata will be written to file.
            data_manager_args (dict): Arguments for the DataManager class instance that will be used to read the data from the .h5 file in outdir and preprocess it.
            n_pca (int): Number of PCA components that will be kept when performing data preprocessing. Defaults to 100.
            plots_dir (str): Directory where the loss curves will be plotted. If None, the plot will not be created. Defaults to None.
            save_preprocessed_data (bool): Whether the preprocessed (i.e. PCA decomposed) training and validation data will be written to file. Defaults to False.
        """

        super().__init__(name = name,
                       outdir = outdir,
                       plots_dir = plots_dir, 
                       save_preprocessed_data = save_preprocessed_data)

        self.n_pca = n_pca

        self.data_manager = DataManager(**data_manager_args)
        self.data_manager.print_file_info()
        self.data_manager.pass_meta_data(self)

        self.preprocess()
        if self.save_preprocessed_data:
            self._save_preprocessed_data()
        
    def preprocess(self):
        """
        Preprocessing method to get the PCA coefficients of the standardized training data.
        It assigns the attributes self.train_X, self.train_y, self.val_X, self.val_y that are passed to the fitting method.
        """
        print(f"Fitting PCA model with {self.n_pca} components to the provided data.")
        self.train_X, self.train_y, self.val_X, self.val_y, self.X_scaler, self.y_scaler = self.data_manager.preprocess_pca(self.n_pca)
        if np.any(np.isnan(self.train_y)) or np.any(np.isnan(self.val_y)):
            raise ValueError(f"Data preprocessing introduced nans. Check raw data for nans of infs or vanishing variance in a specific entry.")
        print(f"PCA model accounts for a share {np.sum(self.y_scaler.explained_variance_ratio_)} of the total variance in the training data. This value is hopefully close to 1.")
        print("Preprocessing data . . . done")
    
    def fit(self,
            config: fiesta_nn.NeuralnetConfig,
            key: jax.random.PRNGKey = jax.random.PRNGKey(0),
            verbose: bool = True):
        """
        Method used to initialize a NN based on the architecture specified in config and then fit it based on the learning rate and epoch number specified in config.
        The config controls which architecture is built through config.hidden_layers.
        
        Args:
            config (fiesta.train.neuralnets.NeuralnetConfig): config that needs to specify at least the network output, hidden_layers, learning rate, and learning epochs. Its output_size must be equal to n_pca.
            key (jax.random.PRNGKey, optional): jax.random.PRNGKey used to initialize the parameters of the network. Defaults to jax.random.PRNGKey(0).
            verbose (bool, optional): Whether the train and validation loss is printed to terminal in certain intervals. Defaults to True.
        """
        
        self.config = config
        self.config.output_size = self.n_pca # the config.output_size has to be equal to the number of PCA components
        input_ndim = len(self.parameter_names)

        
        # Create neural network and initialize the state
        self.network = fiesta_nn.MLP(config = config, input_ndim = input_ndim, key = key)
                
        # Perform training loop
        state, train_losses, val_losses = self.network.train_loop(self.train_X, self.train_y, self.val_X, self.val_y, verbose=verbose)

        # Plot and save the plot if so desired
        if self.plots_dir is not None:
           self.plot_learning_curve(train_losses, val_losses)
       
        self.trained_state = state
        

class CVAETrainer(FluxTrainer):

    def __init__(self,
                 name: str,
                 outdir,
                 data_manager_args,
                 image_size: tuple[Int],
                 plots_dir: str = None,
                 save_preprocessed_data=False)->None:
        """
        FluxTrainer for training a conditional variational autoencoder on the log fluxes of the training data to predict the full 2D spectral flux density array.
        Initializing will read the data and preprocess it with the DataManager class. It can then be fit with the fit() method. 
        To write the surrogate model to file, the save() method is to be used, which will create two pickle files (one for the metadata, one for the neural network).

        Args:
            name (str): Name of the model to be trained. Will be used when saving metadata and model to file.
            outdir (str): Directory where the NN and its metadata will be written to file.
            data_manager_args (dict): Arguments for the DataManager class instance that will be used to read the data from the .h5 file in outdir and preprocess it.
            image_size (tuple(Int)): Size the 2D flux array will be down-sampled to with jax.image.resize when performing data preprocessing.
            plots_dir (str): Directory where the loss curves will be plotted. If None, the plot will not be created. Defaults to None.
            save_preprocessed_data (bool): Whether the preprocessed (i.e. down sampled and standardized) training and validation data will be written to file. Defaults to False.
        """
        
        super().__init__(name = name,
                       outdir = outdir,
                       plots_dir = plots_dir, 
                       save_preprocessed_data = save_preprocessed_data)
        
        self.data_manager = DataManager(**data_manager_args)
        self.data_manager.print_file_info()
        self.data_manager.pass_meta_data(self)

        self.image_size = image_size

        self.preprocess()
        if self.save_preprocessed_data:
            self._save_preprocessed_data()
        
    def preprocess(self)-> None:
        """
        Preprocessing method to get the down_sample arrays of the standardized training data.
        It assigns the attributes self.train_X, self.train_y, self.val_X, self.val_y that are passed to the fitting method.
        """
        print(f"Preprocessing data by resampling flux array to {self.image_size} and standardizing.")
        self.train_X, self.train_y, self.val_X, self.val_y, self.X_scaler, self.y_scaler = self.data_manager.preprocess_cVAE(self.image_size)
        if np.any(np.isnan(self.train_y)) or np.any(np.isnan(self.val_y)):
            raise ValueError(f"Data preprocessing introduced nans. Check raw data for nans of infs or vanishing variance in a specific entry.")
        print("Preprocessing data . . . done")
    
    def fit(self,
            config: fiesta_nn.NeuralnetConfig = None,
            key: jax.random.PRNGKey = jax.random.PRNGKey(0),
            verbose: bool = True) -> None:
        """
        Method used to initialize the autoencoder based on the architecture specified in config and then fit it based on the learning rate and epoch number specified in config.
        The config controls which architecture is built through config.hidden_layers. The encoder and decoder share the hidden_layers argument, though the layers for the decoder are implemented in reverse order.
        
        Args:
            config (fiesta.train.neuralnets.NeuralnetConfig): config that needs to specify at least the network output, hidden_layers, learning rate, and learning epochs. Its output_size must be equal to the product of self.image_size.
            key (jax.random.PRNGKey, optional): jax.random.PRNGKey used to initialize the parameters of the network. Defaults to jax.random.PRNGKey(0).
            verbose (bool, optional): Whether the train and validation loss is printed to terminal in certain intervals. Defaults to True.
        """

        self.config = config
        config.output_size = int(np.prod(self.image_size)) # Output must be equal to the product of self.image_size.

        self.network = fiesta_nn.CVAE(config = self.config, conditional_dim = len(self.parameter_names), key = key)
        state, train_losses, val_losses = self.network.train_loop(self.train_X, self.train_y, self.val_X, self.val_y, verbose = verbose)

        # Plot and save the plot if so desired
        if self.plots_dir is not None:
            self.plot_learning_curve(train_losses, val_losses)     
        
        self.trained_state = state        


###################
# DATA MANAGEMENT #       
###################

class DataManager:
    
    def __init__(self,
                 file: str,
                 n_training: Int,
                 n_val: Int,
                 tmin: Float,
                 tmax: Float,
                 numin: Float = 1e9,
                 numax: Float = 2.5e18,
                 special_training: list = [],
                 ) -> None:
        """
        DataManager class used to handle and preprocess the raw data from the physical model computations stored in an .h5 file.
        Initializing an instance of this class will only read in the meta data, the actual training data and validation data will only be loaded if one of the preprocessing methods is called.

        The .h5 file must contain the following data sets:
            - "times": times in days associated to the spectral flux densities
            - "nus": frequencies in Hz associated to the spectral flux densities
            - "parameter_names": list of the parameter names that are present in the training data.
            - "parameter_distributions": utf-8-string of a dict containing the boundaries and distribution of the parameters.
        Additionally, it must contain three data groups "train", "val", "test". Each of these groups contains two data sets, namely "X" and "y". 
        The X arrays contain the model parameters with columns in the order of "parameter_names" and thus have shape (-1, #parameters). The y array contains the associated log of the spectral flux densities and have shape (-1, #nus * #times).
        To get the full 2D log spectral flux density arrays, one needs to reshape 1D entries of y to (#nus, #times). 
        
        Args:
            file (str): Path to the .h5 file that contains the raw data.
            n_training (int): Number of training data points that will be read in and preprocessed. If used with a FluxTrainer, this is also the number of training data points used to train the model. 
                              Will raise a ValueError, if n_training is larger than the number of training data points stored in the file.
            n_val (int): Number of validation data points that will be read in and preprocessed. If used with a FluxTrainer, this is also the number of validation data points used to monitor the training progress. 
                              Will raise a ValueError, if n_val is larger than the number of validation data points stored in the file.
            tmin (float): Minimum time for which the data will be read in. Fluxes earlier than this time will not be loaded. Defaults to the minimum time of the stored data, if smaller than that value.
            max (float): Maximum time for which the data will be read in. Fluxes later than this time will not be loaded. Defaults to the maximum time of the stored data, if larger than that value.
            numin (float): Minimum frequency for which the data will be read in. Fluxes with frequencies lower than this frequency will not be loaded. Defaults to the minimum frequency of the stored data, if smaller than that value.
            numax (float): Maximum frequency for which the data will be read in. Fluxes with frequencies higher than this frequency will not be loaded. Defaults to the maximum frequency of the stored data, if larger than that value. Defaults to 1e9 Hz (1 GHz).
            special_training (list[str]): Batch of 'special' training data to be added. This can be customly designed training data to cover a certain area of the parameter space more intensily and should be stored in the .h5 file as f['special_train'][label]['X'] and f['special_train'][label]['y'], where label is an entry in this special_training. Defaults to [].
        """
        
        self.file = file
        self.n_training = n_training
        self.n_val = n_val

        self.tmin = tmin
        self.tmax = tmax
        self.numin = numin
        self.numax = numax

        self.special_training = special_training
        
        self.read_metadata_from_file()
        self.set_up_domain_mask()

    def read_metadata_from_file(self,)->None:
        """
        Reads in the metadata of the raw data, i.e., times, frequencies and parameter names. 
        Also determines how many training and validation data points are available.
        """
        with h5py.File(self.file, "r") as f:
            self.times_data = f["times"][:]
            self.nus_data = f["nus"][:]
            self.parameter_names =  f["parameter_names"][:].astype(str).tolist()
            self.n_training_exists = f["train"]["X"].shape[0]
            self.n_val_exists = f["val"]["X"].shape[0]
            self.parameter_distributions = f['parameter_distributions'][()].decode('utf-8')
        
        # check if there is enough data
        if self.n_training > self.n_training_exists: 
            raise ValueError(f"Only {self.n_training_exists} entries in file, not enough to train with {self.n_training} data points.")
        if self.n_val > self.n_val_exists:
                raise ValueError(f"Only {self.n_val_exists} entries in file, not enough to train with {self.n_val} data points.")
    
    def set_up_domain_mask(self,)->None:
        """Trims the stored data down to the time and frequency range desired for training. It sets the mask attribute which is a boolean mask used when loading the data arrays."""
        
        if self.tmin<self.times_data.min() or self.tmax>self.times_data.max():
            print(f"\nWarning: provided time range {self.tmin, self.tmax} is too wide for the data stored in file. Using range {max(self.times_data.min(), self.tmin), min(self.times_data.max(), self.tmax)} instead.\n")
        time_mask = np.logical_and(self.times_data>=self.tmin, self.times_data<=self.tmax)
        self.times = self.times_data[time_mask]
        self.n_times = len(self.times)

        if self.numin<self.nus_data.min() or self.numax>self.nus_data.max():
            print(f"\nWarning: provided frequency range {self.numin, self.numax} is too wide for the data stored in file. Using range {max(self.nus_data.min(), self.numin), min(self.nus_data.max(), self.numax)} instead.\n")
        nu_mask = np.logical_and(self.nus_data>=self.numin, self.nus_data<=self.numax)
        self.nus = self.nus_data[nu_mask]
        self.n_nus = len(self.nus)

        mask = nu_mask[:, None] & time_mask
        self.mask = mask.flatten()
    
    def print_file_info(self,) -> None:
        """
        Prints the meta data of the raw data, i.e., time, frequencies, and parameter names to terminal. 
        Also prints how many training, validation, and test data points are available.
        """
        with h5py.File(self.file, "r") as f:
            print(f"Times: {f['times'][0]} {f['times'][-1]}")
            print(f"Nus: {f['nus'][0]} {f['nus'][-1]}")
            print(f"Parameter distributions: {f['parameter_distributions'][()].decode('utf-8')}")
            print("\n")
            print(f"Training data: {self.n_training_exists}")
            print(f"Validation data: {self.n_val_exists}")
            print(f"Test data: {f['test']['X'].shape[0]}")
            print("Special data:")
            for key in f['special_train'].keys():
                print(f"\t {key}: {f['special_train'][key]['X'].shape[0]}   description: {f['special_train'][key].attrs['comment']}")
            print("\n \n")
    
    def load_raw_data_from_file(self,) -> None:
        """Loads raw data for training and validation as attributes to the instance."""
        with h5py.File(self.file, "r") as f:
            if self.n_training>self.n_training_exists:
                raise ValueError(f"Only {self.n_training_exists} entries in file, not enough to train with {self.n_training} data points.")
            self.train_X_raw = f["train"]["X"][:self.n_training]
            self.train_y_raw = f["train"]["y"][:self.n_training, self.mask]

            for label in self.special_training:
                self.train_X_raw = np.concatenate((self.train_X_raw, f["special_train"][label]["X"][:]))
                self.train_y_raw = np.concatenate((self.train_y_raw, f["special_train"][label]["y"][:, self.mask]))

            if self.n_val>self.n_val_exists:
                raise ValueError(f"Only {self.n_val_exists} entries in file, not enough to validate with {self.n_val} data points.")
            self.val_X_raw = f["val"]["X"][:self.n_val]
            self.val_y_raw = f["val"]["y"][:self.n_val, self.mask]
    
    def preprocess_pca(self, n_components: int) -> tuple[Array, Array, Array, Array, object, object]:
        """
        Loads in the training and validation data and performs PCA decomposition using fiesta.utils.PCADecomposer. 
        Because of memory issues, the training data set is loaded in chunks.
        The X arrays (parameter values) are standardized with fiesta.utils.StandardScalerJax.

        Args:
            n_components(int): Number of PCA components to keep.
        Returns:
            train_X (Array): Standardized training parameters.
            train_y (Array): PCA coefficients of the training data. 
            val_X (Array): Standardized validation parameters
            val_y (Array): PCA coefficients of the validation data.
            Xscaler (StandardScalerJax): Standardizer object fitted to the mean and sigma of the raw training data. Can be used to transform and inverse transform parameter points.
            yscaler (PCAdecomposer): PCAdecomposer object fitted to part of the raw training data. Can be used to transform and inverse transform log spectral flux densities.
        """
        Xscaler, yscaler = StandardScalerJax(), PCAdecomposer(n_components=n_components)
        
        # preprocess the training data
        with h5py.File(self.file, "r") as f:
            train_X_raw = f["train"]["X"][:self.n_training]
            train_X = Xscaler.fit_transform(train_X_raw) # fit the Xscaler and transform the train_X_raw
            
            y_set = f["train"]["y"]

            loaded = y_set[: max(20_000, self.n_training), self.mask].astype(np.float16) # only load max. 20k cause otherwise we might run out of memory at this step
            assert not np.any(np.isinf(loaded)), f"Found inftys in training data."
            yscaler.fit(loaded) # fit the yscaler and transform with the loaded data
            del loaded; gc.collect() # remove loaded from memory

            train_y = np.empty((self.n_training, n_components))

            chunk_size = y_set.chunks[0] # load raw data in chunks of chunk_size
            nchunks, rest = divmod(self.n_training, chunk_size) # load raw data in chunks of chunk_size
            for j, chunk in enumerate(y_set.iter_chunks()):
                loaded = y_set[chunk][:, self.mask]
                assert not np.any(np.isinf(loaded)), f"Found inftys in training data."
                train_y[j*chunk_size:(j+1)*chunk_size] = yscaler.transform(loaded)
                if j>= nchunks-1:
                    break
            if rest > 0:
                loaded = y_set[-rest:, self.mask]
                assert not np.any(np.isinf(loaded)), f"Found inftys in training data."
                train_y[-rest:] = yscaler.transform(loaded)
        
        # preprocess the special training data as well ass the validation data
        train_X, train_y, val_X, val_y = self.__preprocess__special_and_val_data(train_X, train_y, Xscaler, yscaler)

        return train_X, train_y, val_X, val_y, Xscaler, yscaler
    
    def preprocess_cVAE(self, image_size: Int[Array, "shape=(2,)"]) -> tuple[Array, Array, Array, Array, object, object]:
        """
        Loads in the training and validation data and performs data preprocessing for the CVAE using fiesta.utils.ImageScaler. 
        Because of memory issues, the training data set is loaded in chunks.
        The X arrays (parameter values) are standardized with fiesta.utils.StandardScalerJax.

        Args:
            image_size (Array[Int]): Image size the 2D flux arrays are down sampled to with jax.image.resize
        Returns:
            train_X (Array): Standardized training parameters.
            train_y (Array): PCA coefficients of the training data. 
            val_X (Array): Standardized validation parameters
            val_y (Array): PCA coefficients of the validation data.
            Xscaler (StandardScalerJax): Standardizer object fitted to the mean and sigma of the raw training data. Can be used to transform and inverse transform parameter points.
            yscaler (ImageScaler): ImageScaler object fitted to part of the raw training data. Can be used to transform and inverse transform log spectral flux densities.
        """
        Xscaler, yscaler = StandardScalerJax(), ImageScaler(downscale = image_size, upscale = (self.n_nus, self.n_times), scaler = StandardScalerJax())
        
        # preprocess the training data
        with h5py.File(self.file, "r") as f:
            train_X_raw = f["train"]["X"][:self.n_training]
            train_X = Xscaler.fit_transform(train_X_raw) # fit the Xscaler and transform the train_X_raw

            y_set = f["train"]["y"]

            train_y = np.empty((self.n_training, jnp.prod(image_size)), dtype=jnp.float16)
            
            chunk_size = y_set.chunks[0]
            nchunks, rest = divmod(self.n_training, chunk_size) # create raw data in chunks of chunk_size
            for j, chunk in enumerate(y_set.iter_chunks()):
                loaded = y_set[chunk][:, self.mask].astype(jnp.float16)
                assert not np.any(np.isinf(loaded)), f"Found inftys in training data."
                train_y[j*chunk_size:(j+1)*chunk_size] = yscaler.resize_image(loaded).reshape(-1, jnp.prod(image_size))
                if j>= nchunks-1:
                    break
            if rest > 0:
                loaded = y_set[-rest:, self.mask].astype(jnp.float16)
                assert not np.any(np.isinf(loaded)), f"Found inftys in training data."
                train_y[-rest:] = yscaler.resize_image(loaded).reshape(-1, jnp.prod(image_size))
            
            train_y = yscaler.fit_transform_scaler(train_y) # this standardizes now the down sampled fluxes

        # preprocess the special training data as well ass the validation data
        train_X, train_y, val_X, val_y = self.__preprocess__special_and_val_data(train_X, train_y, Xscaler, yscaler)
        return train_X, train_y, val_X, val_y, Xscaler, yscaler
    

    def __preprocess__special_and_val_data(self, train_X, train_y, Xscaler, yscaler) -> tuple[Array, Array, Array, Array]:
        """ sub method that just applies the scaling transforms to the validation and special training data """
        with h5py.File(self.file, "r") as f:
            # preprocess the special training data       
            for label in self.special_training:
                special_train_x = Xscaler.transform(f["special_train"][label]["X"][:])
                train_X = np.concatenate((train_X, special_train_x))

                special_train_y = yscaler.transform(f["special_train"][label]["y"][:, self.mask])
                train_y = np.concatenate(( train_y, special_train_y.astype(jnp.float16) ))

            # preprocess validation data
            val_X_raw = f["val"]["X"][:self.n_val]
            val_X = Xscaler.transform(val_X_raw)
            val_y_raw = f["val"]["y"][:self.n_val, self.mask]
            val_y = yscaler.transform(val_y_raw)
        
        return train_X, train_y, val_X, val_y
    
    def pass_meta_data(self, object) -> None:
        """Pass training data meta data to another object. Used for the FluxTrainers."""
        object.parameter_names = self.parameter_names
        object.times = self.times
        object.nus = self.nus
        object.parameter_distributions = self.parameter_distributions