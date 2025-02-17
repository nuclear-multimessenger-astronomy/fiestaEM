"""Method to train the surrogate models"""

import fiesta.train.neuralnets as fiesta_nn
from fiesta.train.DataManager import DataManager

import os
import numpy as np

import jax
from jaxtyping import Array, Float, Int

import matplotlib.pyplot as plt
import pickle

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
        plt.ylabel("Loss")
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
        save["model_type"] = self.model_type

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
        
        self.model_type = "MLP"

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
        
        self.model_type = "CVAE"
        
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
