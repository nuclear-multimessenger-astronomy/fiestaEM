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
    """Abstract class for training a surrogate model that predicts a flux array."""
    
    name: str
    outdir: str
    parameter_names: list[str]

    train_X: Float[Array, "n_train"]
    train_y: Float[Array, "n_train"]
    val_X: Float[Array, "n_val"]
    val_y: Float[Array, "n_val"]
    
    trained_states: dict[str, fiesta_nn.TrainState]
    
    def __init__(self, 
                 name: str,
                 outdir: str,
                 plots_dir: str = None,
                 save_preprocessed_data: bool = False
                 ) -> None:
        
        self.name = name
        self.outdir = outdir
        # Check if directories exists, otherwise, create:
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        
        self.plots_dir = plots_dir
        if self.plots_dir is not None and not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
        
       
        # To be loaded by child classes
        self.parameter_names = None
        self.save_preprocessed_data = save_preprocessed_data
        
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
            verbose: bool = True):
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
    
    def save(self):
        """
        Save the trained model and all the used metadata to the outdir.
        """
        # Save the metadata
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
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
    
    def _save_preprocessed_data(self):
        print("Saving preprocessed data . . .")
        np.savez(os.path.join(self.outdir, "afterglow_preprocessed_data.npz"), train_X=self.train_X, train_y= self.train_y, val_X = self.val_X, val_y = self.val_y)
        print("Saving preprocessed data . . . done")

class PCATrainer(FluxTrainer):

    def __init__(self,
                 name: str,
                 outdir: str,
                 data_manager_args: dict,
                 n_pca: Int = 100,
                 plots_dir: str = None,
                 save_preprocessed_data: bool = False):

        super().__init__(name = name,
                       outdir = outdir,
                       plots_dir = plots_dir, 
                       save_preprocessed_data = save_preprocessed_data)

        self.n_pca = n_pca

        self.data_manager = DataManager(**data_manager_args)
        self.data_manager.print_file_info()
        self.data_manager._pass_meta_data(self)

        self.preprocess()
        if self.save_preprocessed_data:
            self._save_preprocessed_data()
        
    def preprocess(self):
        print(f"Fitting PCA model with {self.n_pca} components to the provided data.")
        self.train_X, self.train_y, self.val_X, self.val_y, self.X_scaler, self.y_scaler = self.data_manager.preprocess_pca(self.n_pca)
        print(f"PCA model accounts for a share {np.sum(self.y_scaler.explained_variance_ratio_)} of the total variance in the training data. This value is hopefully close to 1.")
        print("Preprocessing data . . . done")
    
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
                 plots_dir: str =f"./benchmarks/",
                 save_preprocessed_data=False)->None:
        
        super().__init__(name = name,
                       outdir = outdir,
                       plots_dir = plots_dir, 
                       save_preprocessed_data = save_preprocessed_data)
        
        self.data_manager = DataManager(**data_manager_args)
        self.data_manager.print_file_info()
        self.data_manager._pass_meta_data(self)

        self.image_size = image_size

        self.preprocess()
        if self.save_preprocessed_data:
            self._save_preprocessed_data()
        
    def preprocess(self)-> None:
        print(f"Preprocessing data by resampling flux array to {self.image_size} and standardizing.")
        self.train_X, self.train_y, self.val_X, self.val_y, self.X_scaler, self.y_scaler = self.data_manager.preprocess_cVAE(self.image_size)
        print("Preprocessing data . . . done")
    
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
                 ):
        
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
        with h5py.File(self.file, "r") as f:
            self.times_data = f["times"][:]
            self.nus_data = f["nus"][:]
            self.parameter_names =  f["parameter_names"][:].astype(str).tolist()
            self.n_training_exists = f["train"]["X"].shape[0]
            self.n_val_exists = f["val"]["X"].shape[0]
            self.parameter_distributions = f['parameter_distributions'][()].decode('utf-8')
    
    def set_up_domain_mask(self,)->None:
        """Trims the stored data down to the time and frequency range desired for training."""
        
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
    
    def load_raw_data_from_file(self,):
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
    
    def preprocess_pca(self, n_components: int)->None:
        Xscaler, yscaler = StandardScalerJax(), PCAdecomposer(n_components=n_components)
        if self.n_training > self.n_training_exists: # check if there is enough data
            raise ValueError(f"Only {self.n_training_exists} entries in file, not enough to train with {self.n_training} data points.")
        if self.n_val > self.n_val_exists:
                raise ValueError(f"Only {self.n_val_exists} entries in file, not enough to train with {self.n_val} data points.")
        
        # preprocess the training data
        with h5py.File(self.file, "r") as f:
            train_X_raw = f["train"]["X"][:self.n_training]
            train_X = Xscaler.fit_transform(train_X_raw)
            
            y_set = f["train"]["y"]
            chunk_size = y_set.chunks[0]

            loaded = y_set[:20_000, self.mask].astype(np.float16) # only load 20k cause otherwise we might run out of memory at this step
            if np.any(np.isinf(loaded)):
                raise ValueError(f"Found inftys in training data.")
            yscaler.fit(loaded); del loaded; gc.collect() # remove loaded from memory

            train_y = np.empty((self.n_training, n_components))
            nchunks, rest = divmod(self.n_training, chunk_size) # load raw data in chunks of chunk_size
            for j, chunk in enumerate(y_set.iter_chunks()):
                loaded = y_set[chunk][:, self.mask]
                if np.any(np.isinf(loaded)):
                    raise ValueError(f"Found inftys in training data.")
                train_y[j*chunk_size:(j+1)*chunk_size] = yscaler.transform(loaded)
                if j>= nchunks-1:
                    break
            if rest > 0:
                loaded = y_set[-rest:, self.mask]
                if np.any(np.isinf(loaded)):
                    raise ValueError(f"Found inftys in training data.")
                train_y[-rest:] = yscaler.transform(loaded)
        
        # preprocess the special training data as well ass the validation data
        train_X, train_y, val_X, val_y = self.__preprocess__special_and_val_data(train_X, train_y, Xscaler, yscaler)

        return train_X, train_y, val_X, val_y, Xscaler, yscaler
    
    def preprocess_cVAE(self, image_size: Int[Array, "shape=(2,)"]):
        Xscaler, yscaler = MinMaxScalerJax(), ImageScaler(downscale = image_size, upscale = (self.n_nus, self.n_times))
        if self.n_training > self.n_training_exists: # check if there is enough data
            raise ValueError(f"Only {self.n_training_exists} entries in file, not enough to train with {self.n_training} data points.")
        if self.n_val > self.n_val_exists:
                raise ValueError(f"Only {self.n_val_exists} entries in file, not enough to train with {self.n_val} data points.")
        
        # preprocess the training data
        with h5py.File(self.file, "r") as f:
            train_X_raw = f["train"]["X"][:self.n_training]
            train_X = Xscaler.fit_transform(train_X_raw)

            y_set = f["train"]["y"]
            chunk_size = y_set.chunks[0]
            test = y_set[0]

            train_y = np.empty((self.n_training, jnp.prod(image_size)), dtype=jnp.float16)
            nchunks, rest = divmod(self.n_training, chunk_size) # create raw data in chunks of chunk_size
            for j, chunk in enumerate(y_set.iter_chunks()):
                loaded = y_set[chunk][:, self.mask].astype(jnp.float16).reshape(-1, self.n_nus, self.n_times)
                if np.any(np.isinf(loaded)):
                    raise ValueError(f"Found inftys in training data.")
                train_y[j*chunk_size:(j+1)*chunk_size] = yscaler.resize_image(loaded).reshape(-1, jnp.prod(image_size))
                if j>= nchunks-1:
                    break
            if rest > 0:
                loaded = y_set[-rest:, self.mask].astype(jnp.float16).reshape(-1, self.n_nus, self.n_times)
                if np.any(np.isinf(loaded)):
                    raise ValueError(f"Found inftys in training data.")
                train_y[-rest:] = yscaler.resize_image(loaded)
            
            standardscaler = StandardScalerJax()
            train_y = standardscaler.fit_transform(train_y)
            yscaler.mu, yscaler.sigma = standardscaler.mu, standardscaler.sigma # a bit hacky here, but that is because of loading the data in chunks above

        # preprocess the special training data as well ass the validation data
        train_X, train_y, val_X, val_y = self.__preprocess__special_and_val_data(train_X, train_y, Xscaler, yscaler)
        return train_X, train_y, val_X, val_y, Xscaler, yscaler
    

    def __preprocess__special_and_val_data(self, train_X, train_y, Xscaler, yscaler):
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
    
    def _pass_meta_data(self, object):
        object.parameter_names = self.parameter_names
        object.times = self.times
        object.nus = self.nus
        object.parameter_distributions = self.parameter_distributions
    

    def print_file_info(self,):
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