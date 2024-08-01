import os
from typing import Sequence, Callable
import functools
import time
import numpy as np

import jax
import jax.numpy as jnp

from jaxtyping import Array, Float

import flax
from flax import linen as nn  # Linen API
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
from flax.training.train_state import TrainState

from ml_collections import ConfigDict
import optax

import pickle


###############
### CONFIGS ###
###############

class NeuralnetConfig(ConfigDict):
    """Configuration for a neural network model. For type hinting"""
    name: str = "MLP"
    layer_sizes: Sequence[int] = [64, 128, 64, 10]
    act_func: Callable = nn.relu
    optimizer: Callable = optax.adam
    learning_rate: float = 1e-3
    batch_size: int = 128
    nb_epochs: int = 1000
    nb_report: int = 100
    
    # TODO: add support for schedulers in the future
    
    # fixed_lr: bool
    # my_scheduler: ConfigDict
    # nb_epochs_decay: int
    # learning_rate_fn: Callable

    # For the scheduler:
    # config.nb_epochs_decay = int(round(config.nb_epochs / 10))
    # config.learning_rate_fn = None
    # ^ optax learning rate scheduler
    # # Custom scheduler (work in progress...)
    # config.fixed_lr = False
    # config.my_scheduler = ConfigDict() # to gather parameters
    # # In case of false fixed learning rate, will adapt lr based on following params for custom scheduler
    # config.my_scheduler.counter = 0
    # # ^ count epochs during training loop, in order to only reduce lr after x amount of steps
    # config.my_scheduler.threshold = 0.995
    # # ^ if best loss has not recently improved by this fraction, then reduce learning rate
    # config.my_scheduler.multiplier = 0.5
    # # ^ reduce lr by this factor if loss is not improving sufficiently
    # config.my_scheduler.patience = 10
    # # ^ amount of epochs to wait in loss curve before adapting lr if loss goes up
    # # config.my_scheduler.burnin = 20
    # # # ^ amount of epochs to wait at start before any adaptation is done
    # config.my_scheduler.history = 10
    # # ^ amount of epochs to "look back" in order to determine whether loss is improving or not

#####################
### ARCHITECTURES ###
#####################

class BaseNeuralnet(nn.Module):
    """Abstract base class. Needs layer sizes and activation function used"""
    layer_sizes: Sequence[int]
    act_func: Callable
    
    def __init__(self, 
                 layer_sizes: Sequence[int],
                 act_func: Callable):
        """
        Initialize the neural network with the given layer sizes and activation function.

        Args:
            layer_sizes (Sequence[int]): List of integers representing the number of neurons in each layer.
            act_func (Callable): Activation function to be used in the network.
        """
        
        assert len(layer_sizes) > 1, "Need at least two layers for a neural network"
        self.layer_sizes = layer_sizes
        self.act_func = act_func

    def setup(self):
        raise NotImplementedError
    
    def __call__(self, x):
        raise NotImplementedError    

class MLP(BaseNeuralnet):
    """Basic multi-layer perceptron: a feedforward neural network with multiple Dense layers."""

    def __init__(self, 
                 layer_sizes: Sequence[int],
                 act_func: Callable):
        super().__init__(layer_sizes, act_func)

    def setup(self):
        self.layers = [nn.Dense(n) for n in self.layer_sizes]

    # TODO: to jit or not to jit?
    # @functools.partial(jax.jit, static_argnums=(2, 3))
    @nn.compact
    def __call__(self, x: Array):
        """_summary_

        Args:
            x (Array): Input data of the neural network.
        """

        for i, layer in enumerate(self.layers):
            # Apply the linear part of the layer's operation
            x = layer(x)
            # If not the output layer, apply the given activation function
            if i != len(self.layer_sizes) - 1:
                x = self.act_func(x)

        return x


# TODO: can this be removed now?
# class NeuralNetwork(nn.Module):
#     """A very basic initial neural network used for testing the basic functionalities of Flax.

#     Returns:
#         NeuralNetwork: The architecture of the neural network
#     """

#     @nn.compact
#     def __call__(self, x):
#         x = nn.Dense(features=24)(x)
#         x = nn.relu(x)
#         x = nn.Dense(features=64)(x)
#         x = nn.relu(x)
#         x = nn.Dense(features=24)(x)
#         x = nn.relu(x)
#         x = nn.Dense(features=10)(x)
#         return x


################
### TRAINING ###
################

def my_lr_scheduler(config: NeuralnetConfig, state, train_losses) -> None:
    """Custom learning rate scheduler

    Args:
        config (ConfigDict): Configuration dict for experiments.
        train_losses (list): Train losses recorded so far during a training loop.

    Returns:
        None
    """

    # If fixed lr, then no scheduler: return old state again
    if config.fixed_lr:
        return state

    # Defined custom scheduler here. Compare history of loss curve to previous best
    patience = config.my_scheduler.patience
    new_state = state
    if config.my_scheduler.counter >= patience:
        print(train_losses)
        print(train_losses[-patience // 2:])
        print(train_losses[-patience : -patience // 2])
        current_best = jnp.min(train_losses[-patience // 2:])
        previous_best = jnp.min(train_losses[-patience:-patience // 2])

        # If we did not improve the test loss sufficiently, going to adapt LR
        if current_best / previous_best >= config.my_scheduler.threshold:
            # Reset counter (note: will increment later, so set to -1 st it becomes 0)
            config.my_scheduler.counter = -1
            config.learning_rate = config.my_scheduler.multiplier * config.learning_rate
            # Reset optimizer
            tx = config.optimizer(learning_rate = config.learning_rate)
            new_state = TrainState.create(apply_fn = state.apply_fn, params = state.params, tx = tx)

    # Add to epoch counter for the scheduler
    config.my_scheduler.counter += 1
    return new_state



def create_train_state(model, test_input, rng, config):
    """
    Creates an initial `TrainState` from NN model and optimizer. Test input and RNG for initialization of the NN.
    TODO add Optax scheduler possibility here
    """
    # Initialize the parameters by passing dummy input
    params = model.init(rng, test_input)['params']
    tx = config.optimizer(config.learning_rate)
    state = TrainState.create(apply_fn = model.apply, params = params, tx = tx)
    return state

def apply_model(state, x_batched, y_batched):

    def loss_fn(params):
        def squared_error(x, y):
            # For a single datapoint
            pred = state.apply_fn({'params': params}, x)
            return jnp.inner(y - pred, y - pred) / 2.0
        # Vectorize the previous to compute the average of the loss on all samples.
        return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return loss, grads

@jax.jit
def train_step(state, train_X, train_y, val_X = None, val_y = None):
    """
    Train for a single step. Note that this function is functionally pure and hence suitable for jit.
    """

    # Compute losses
    train_loss, grads = apply_model(state, train_X, train_y)
    if val_X is not None:
        val_loss, grads = apply_model(state, val_X, val_y)

    # Update parameters
    state = state.apply_gradients(grads=grads)

    return state, train_loss, val_loss

# @jax.jit
# TODO replace jit?
# TODO use tqdm?
def train_loop(state: TrainState, train_X, train_y, val_X = None, val_y = None, config = None):

    train_losses, val_losses = [], []

    if config is None:
        config = get_default_config()

    start = time.time()
    for i in range(config.nb_epochs):
        # Do a single step
        state, train_loss, val_loss = train_step(state, train_X, train_y, val_X, val_y)
        # Save the losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # Report once in a while
        if i % config.nb_report == 0:
            print(f"Train loss at step {i+1}: {train_loss}")
            print(f"Valid loss at step {i+1}: {val_loss}")
            print(f"Learning rate: {config.learning_rate}")
            print("---")
        # TODO add my custom scheduler here?

    end = time.time()
    print(f"Training for {config.nb_epochs} took {end-start} seconds.")

    return state, train_losses, val_losses

def serialize(state: TrainState, config: ConfigDict = None):
    # Create own serialization to save later on
    
    # If no config dict was given, we assume we used the default one
    if config is None:
        config = get_default_config()
    
    # Get state dict, which has params
    params = flax.serialization.to_state_dict(state)["params"]
    
    # TODO why is act func throwing errors?
    # Quick workaround:
    del config["act_func"]
    
    serialized_dict = {"params": params,
                    "config": config,
                    }
    
    return serialized_dict

# TODO improve documentation below

# TODO add support for various activation functions and model architectures to be loaded
def save_model(state: TrainState, config: ConfigDict = None, out_name: str = "my_flax_model.pkl"):
    serialized_dict = serialize(state, config)
    with open(out_name, 'wb') as handle:
        pickle.dump(serialized_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
def load_model(filename):
    """Load and return training state for a single model"""
    # TODO get model architecture from a string
    
    with open(filename, 'rb') as handle:
        loaded_dict = pickle.load(handle)
        
    config = loaded_dict["config"]
    layer_sizes = config["layer_sizes"]
    # TODO this throws errors
    act_func = flax.linen.relu
    # act_func = config["act_func"]
    params = loaded_dict["params"]
        
    if config["name"] == "MLP":
        model = MLP(layer_sizes, act_func)
    else:
        raise ValueError("Error loading model, architecture name not recognized.")
    
    # # Initialize train state
    # # TODO cumbersome way to fetch the input dimension, is there a better way? I.e. save input ndim while saving model?
    # params_keys = list(params.keys())
    # first_layer = params[params_keys[0]]
    # input_ndim = np.shape(first_layer)[0]
    
    # Create train state without optimizer
    state = TrainState.create(apply_fn = model.apply, params = params, tx = optax.adam(config.learning_rate))
    
    return state

def save_model_all_filts(svd_model: SVDTrainingModel, config: ConfigDict = None, out_name: str = "my_flax_model"):
    # Save the learned model for all filters in SVD model
    filters = list(svd_model.keys())
    for filt in filters:
        model = svd_model[filt]["model"]
        save_model(model, config, out_name=out_name + f"_{filt}.pkl")
        
def load_model_all_filts(svd_model: SVDTrainingModel, model_dir: str):
    # Iterate over all the filters that are present in the SVD model
    filters = list(svd_model.keys())
    for filt in filters:
        # Check whether we have a saved model for this filter
        # TODO what if file extension changes?
        filenames = [file for file in os.listdir(model_dir) if f"{filt}.pkl" in file]
        if len(filenames) == 0:
            raise ValueError(f"Error loading flax model: filter {filt} does not seem to be saved in directory {model_dir}")
        elif len(filenames) > 1:
            print(f"Warning: there are several matches with filter {filt} in directory {model_dir}, loading first")
        # If we have a saved model, load in and save into our object
        filename = filenames[0]
        state = load_model(model_dir + filename)
        svd_model[filt]["model"] = state