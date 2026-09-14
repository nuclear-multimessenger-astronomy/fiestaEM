"""Utils for dealing with the neural networks"""

from jaxtyping import Float, Int, Array
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState
from ml_collections import ConfigDict




###############
### CONFIGS ###
###############

class NeuralnetConfig(ConfigDict):
    """Configuration for a neural network model. For type hinting"""
    name: str
    input_size: Int
    output_size: Int
    hidden_layer_sizes: list[int]
    learning_rate: Float

    
    def __init__(
            self,
            name: str = "MLP",
            output_size: int = 10,
            input_size: int = 10,
            hidden_layer_sizes: list[int] = [64, 128, 64],
            learning_rate: Float = 1e-3,
            latent_dim: int = 20,
            weight_decay: Float = 0.0,
            batch_size: int = 128,
            nb_epochs: Int = 1_000,
            nb_report: Int = None,
            dropout_rate: float = 0.0,
            use_cosine_schedule: bool = False,
            cosine_alpha: float = 0.01,
            max_grad_norm: float = 0.0,
            pca_smoothness_weight: float = 0.0,
            pca_smoothness_start: int = 0
        ):

        super().__init__()
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layer_sizes = [*hidden_layer_sizes, output_size]
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        if nb_report is None:
            nb_report = max(1, self.nb_epochs // 10)
        self.nb_report = nb_report
        self.dropout_rate = dropout_rate
        self.use_cosine_schedule = use_cosine_schedule
        self.cosine_alpha = cosine_alpha
        self.max_grad_norm = max_grad_norm
        self.pca_smoothness_weight = pca_smoothness_weight
        self.pca_smoothness_start = pca_smoothness_start

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