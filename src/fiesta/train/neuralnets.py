import time

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

import flax
from flax import linen as nn  # Linen API
from flax.training.train_state import TrainState
from ml_collections import ConfigDict
import optax
import pickle

import fiesta.train.nn_architectures as nn
from fiesta.logging import logger

###############
### CONFIGS ###
###############

class NeuralnetConfig(ConfigDict):
    """Configuration for a neural network model. For type hinting"""
    name: str
    output_size: Int
    hidden_layer_sizes: list[int]
    layer_sizes: list[int]
    latent_dim: Int
    learning_rate: Float
    batch_size: Int
    nb_epochs: Int
    nb_report: Int
    
    def __init__(self,
                 name: str = "MLP",
                 output_size: int = 10,
                 hidden_layer_sizes: list[int] = [64, 128, 64],
                 latent_dim: int = 20,
                 learning_rate: Float = 1e-3,
                 batch_size: int = 128,
                 nb_epochs: Int = 1_000,
                 nb_report: Int = None):
        
        super().__init__()
        self.name = name
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layer_sizes = [*hidden_layer_sizes, output_size]
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        if nb_report is None:
            nb_report = self.nb_epochs // 10
        self.nb_report = nb_report

#############
### UTILS ###
#############

def kld(mean, logvar):
    """
    Kullback-Leibler divergence of a normal distribution with arbitrary mean and log variance to the standard normal distribution with mean 0 and unit variance.
    """
    return 0.5 * jnp.sum(mean**2 + jnp.exp(logvar) - logvar -1)

def bce(y, pred):
    """
    binary cross entropy between y and the predicted array pred
    """
    return -jnp.sum(y * jnp.log(pred) + (1-y) * jnp.log(1-pred))

def mse(y, pred):
    """
    square error between y and the predicted array pred
    """
    return jnp.sum((y-pred)**2)

def serialize(state: TrainState, 
              config: NeuralnetConfig = None) -> dict:
    """
    Serialize function to save the model and its configuration.

    Args:
        state (TrainState): The TrainState object to be serialized.
        config (NeuralnetConfig, optional): The config to be serialized. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    # Get state dict, which has params
    params = flax.serialization.to_state_dict(state)["params"]
    
    serialized_dict = {"params": params,
                       "config": config}
    
    return serialized_dict

################
### TRAINING ###
################


class CVAE:
    def __init__(self,
                 config: NeuralnetConfig,
                 conditional_dim: Int,
                 key: jax.random.PRNGKey = jax.random.key(21)):
        self.config = config
        net = nn.CVAE(hidden_layer_sizes=config.hidden_layer_sizes, latent_dim=config.latent_dim, output_size=config.output_size)
        key, subkey, subkey2 = jax.random.split(key, 3)

        params = net.init(subkey, jnp.ones(config.output_size), jnp.ones(conditional_dim), subkey2)['params']
        tx = optax.adam(config.learning_rate)
        self.state = TrainState.create(apply_fn = net.apply, params = params, tx = tx) # initialize the training state
    
    @staticmethod
    @jax.jit
    def train_step(state: TrainState, 
                   train_X: Float[Array, "n_batch_train ndim_input"], 
                   train_y: Float[Array, "n_batch_train ndim_output"],
                   rng: jax.random.PRNGKey,
                   val_X: Float[Array, "n_batch_val ndim_output"] = None, 
                   val_y: Float[Array, "n_batch_val ndim_output"] = None, 
                   ) -> tuple[TrainState, Float[Array, "n_batch_train"], Float[Array, "n_batch_val"]]:
        def apply_model(state, X, y, z_rng):
            def loss_fn(params):
                reconstructed_y, mean, logvar = state.apply_fn({'params': params}, y, X, z_rng)
                mse_loss =  jnp.mean(jax.vmap(mse)(y, reconstructed_y)) # mean squared error loss
                kld_loss = jnp.mean(jax.vmap(kld)(mean, logvar)) # KLD loss
                return mse_loss + kld_loss
    
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            return loss, grads
        rng, z_rng = jax.random.split(rng)
        train_loss, grads = apply_model(state, train_X, train_y, z_rng)
        if val_X is not None:
            rng, z_rng = jax.random.split(rng)
            val_loss, _ = apply_model(state, val_X, val_y, z_rng)
        else:
            val_loss = jnp.zeros_like(train_loss)
    
        # Update parameters
        state = state.apply_gradients(grads=grads)
    
        return state, train_loss, val_loss, rng
    
    def train_loop(self,
                   train_X: Float[Array, "n_batch_train ndim_input"], 
                   train_y: Float[Array, "n_batch_train ndim_output"],
                   val_X: Float[Array, "n_batch_val ndim_output"] = None, 
                   val_y: Float[Array, "n_batch_val ndim_output"] = None,
                   verbose: bool = True):
    
        train_losses, val_losses = [], []
        rng = jax.random.key(2025)
        state = self.state
    
        start = time.time()
        
        for i in range(self.config.nb_epochs):
            # Do a single step
            rng, subkey = jax.random.split(rng)
            state, train_loss, val_loss, rng = self.train_step(state, train_X, train_y, subkey, val_X, val_y)
            # Save the losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            # Report once in a while
            if i % self.config.nb_report == 0 and verbose:
                logger.info(f"Train loss at step {i+1}: {train_loss}")
                logger.info(f"Valid loss at step {i+1}: {val_loss}")
                logger.info(f"Learning rate: {self.config.learning_rate}")
                logger.info("---")
    
        end = time.time()
        if verbose:
            logger.info(f"Training for {self.config.nb_epochs} took {end-start} seconds.")
        
        self.trained_state = state
    
        return self.trained_state, train_losses, val_losses
    
    def save_model(self, outfile: str = "my_flax_model.pkl"):
        """
        Serialize and save the model to a file.
        
        Raises:
            ValueError: If the provided file extension is not .pkl or .pickle.
    
        Args:
            outfile (str, optional): The pickle file to which we save the serialized model. Defaults to "my_flax_model.pkl".
        """
        
        if not outfile.endswith(".pkl") and not outfile.endswith(".pickle"):
            raise ValueError("For now, only .pkl or .pickle extensions are supported.")
        
        serialized_dict = serialize(self.trained_state, self.config)
        with open(outfile, 'wb') as handle:
            pickle.dump(serialized_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_model(filename: str) -> tuple[TrainState, NeuralnetConfig]:
        """
        Load a model from a file.
        TODO: this is very cumbersome now and must be massively improved in the future
    
        Args:
            filename (str): Filename of the model to be loaded.
    
        Raises:
            ValueError: If there is something wrong with loading, since lots of things can go wrong here.
    
        Returns:
            tuple[TrainState, NeuralnetConfig]: The TrainState object loaded from the file and the NeuralnetConfig object.
        """
        with open(filename, 'rb') as handle:
            loaded_dict = pickle.load(handle)
            
        config: NeuralnetConfig = loaded_dict["config"]
        params = loaded_dict["params"]

        net = nn.Decoder(layer_sizes = [*config.hidden_layer_sizes[::-1], config.output_size])
        # Create train state without optimizer
        state = TrainState.create(apply_fn = net.apply, params = params["decoder"], tx = optax.adam(config.learning_rate))
        
        return state, config
    
    @staticmethod
    def load_full_model(filename: str) -> tuple[TrainState, NeuralnetConfig]:

        with open(filename, "rb") as handle:
            loaded_dict = pickle.load(handle)         
        
        config: NeuralnetConfig = loaded_dict["config"]
        params = loaded_dict["params"]

        net = nn.CVAE(hidden_layer_sizes=config.hidden_layer_sizes, output_size= config.output_size)
        # Create train state without optimizer
        state = TrainState.create(apply_fn = net.apply, params = params, tx = optax.adam(config.learning_rate))

        return state, config
        

class MLP:
    def __init__(self,
                 config: NeuralnetConfig,
                 input_ndim: Int,
                 key: jax.random.PRNGKey = jax.random.key(21)):
        self.config = config
        net = nn.MLP(layer_sizes= config.layer_sizes)
        key, subkey = jax.random.split(key)
        params = net.init(subkey, jnp.ones(input_ndim))['params']
        tx = optax.adam(config.learning_rate)
        self.state = TrainState.create(apply_fn = net.apply, params = params, tx = tx) # initialize the training state

    @staticmethod
    @jax.jit
    def train_step(state: TrainState, 
                   train_X: Float[Array, "n_batch_train ndim_input"], 
                   train_y: Float[Array, "n_batch_train ndim_output"],
                   val_X: Float[Array, "n_batch_val ndim_output"] = None, 
                   val_y: Float[Array, "n_batch_val ndim_output"] = None, 
                   ) -> tuple[TrainState, Float[Array, "n_batch_train"], Float[Array, "n_batch_val"]]:
        def apply_model(state, X, y):
            def loss_fn(params):
                reconstructed_y = state.apply_fn({'params': params}, X)
                mse_loss =  jnp.mean(jax.vmap(mse)(y, reconstructed_y)) # mean squared error loss
                return mse_loss
    
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            return loss, grads
        train_loss, grads = apply_model(state, train_X, train_y)
        if val_X is not None:
            val_loss, _ = apply_model(state, val_X, val_y)
        else:
            val_loss = jnp.zeros_like(train_loss)
    
        # Update parameters
        state = state.apply_gradients(grads=grads)
    
        return state, train_loss, val_loss
    
    def train_loop(self,
                   train_X: Float[Array, "n_batch_train ndim_input"], 
                   train_y: Float[Array, "n_batch_train ndim_output"],
                   val_X: Float[Array, "n_batch_val ndim_output"] = None, 
                   val_y: Float[Array, "n_batch_val ndim_output"] = None,
                   verbose: bool = True):
    
        train_losses, val_losses = [], []
        state = self.state
    
        start = time.time()
        
        for i in range(self.config.nb_epochs):
            # Do a single step
            state, train_loss, val_loss = self.train_step(state, train_X, train_y, val_X, val_y)
            # Save the losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            # Report once in a while
            if i % self.config.nb_report == 0 and verbose:
                logger.info(f"Train loss at step {i+1}: {train_loss}")
                logger.info(f"Valid loss at step {i+1}: {val_loss}")
                logger.info(f"Learning rate: {self.config.learning_rate}")
                logger.info("---")
    
        end = time.time()
        if verbose:
            logger.info(f"Training for {self.config.nb_epochs} took {end-start} seconds.")
        
        self.trained_state = state
    
        return self.trained_state, train_losses, val_losses
    
    def save_model(self, outfile: str = "my_flax_model.pkl"):
        """
        Serialize and save the model to a file.
        
        Raises:
            ValueError: If the provided file extension is not .pkl or .pickle.
    
        Args:
            outfile (str, optional): The pickle file to which we save the serialized model. Defaults to "my_flax_model.pkl".
        """
        
        if not outfile.endswith(".pkl") and not outfile.endswith(".pickle"):
            raise ValueError("For now, only .pkl or .pickle extensions are supported.")
        
        serialized_dict = serialize(self.trained_state, self.config)
        with open(outfile, 'wb') as handle:
            pickle.dump(serialized_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_model(filename: str) -> tuple[TrainState, NeuralnetConfig]:
        """
        Load a model from a file.
        TODO: this is very cumbersome now and must be massively improved in the future
    
        Args:
            filename (str): Filename of the model to be loaded.
    
        Raises:
            ValueError: If there is something wrong with loading, since lots of things can go wrong here.
    
        Returns:
            tuple[TrainState, NeuralnetConfig]: The TrainState object loaded from the file and the NeuralnetConfig object.
        """
        with open(filename, 'rb') as handle:
            loaded_dict = pickle.load(handle)

        config: NeuralnetConfig = loaded_dict["config"]
        params = loaded_dict["params"]
            
        net = nn.MLP(config.layer_sizes)
        # Create train state without optimizer
        state = TrainState.create(apply_fn = net.apply, params = params, tx = optax.adam(config.learning_rate))
    
        return state, config            