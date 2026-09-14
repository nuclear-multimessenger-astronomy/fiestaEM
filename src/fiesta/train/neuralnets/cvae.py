"""Class for the conditional variational autoencoder."""
import pickle
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float
from flax.training.train_state import TrainState

import fiesta.train.nn_architectures as nn
from fiesta.logging import logger
from fiesta.train import DataLoader
from fiesta.scalers import (
    ParameterScaler,
    DataScaler,
    StandardScalerJax,
    ImageScaler
)

from .base import NN
from .utils import NeuralnetConfig, mse, kld

class CVAE(NN):
    """
    Conditional variational autoencoder using the flax-interface.

    Args: 
        config (NeuralnetConfig): NN config dictionary. 
                                  Its ``latent_dim`` will determine the size of the latent layer.                   
        image_size (tuple[int]): Tuple of length two that will determine to which size the 2D arrays
                                 for the flux densities are down scaled to when preprocessing the data.
                                 This also then becomes the input and output dimension of the CVAE.
        key (PRNGKey, optional): Random key for initialization. Defaults to ``21``.
    """
    def __init__(
            self,
            config: NeuralnetConfig,
            image_size: tuple[int], 
            key: jax.random.PRNGKey = jax.random.key(21)
        ):

        self.config = config
        if len(image_size) !=2:
            raise ValueError("``image_size`` must be a tuple of length 2.")
        # FIXME: image_size here must be a numpy-array
        # so that the ImageScaler does not break during pickling.
        # In future the ImageScaler should just store the image sizes as attributes.
        self.image_size = np.array(image_size)
        self.input_size = int(np.prod(image_size))
        self.output_size = self.input_size
        key, subkey = jax.random.split(key)

        
        # Create the neural network
        net = nn.CVAE(
            hidden_layer_sizes=config.hidden_layer_sizes, 
            latent_dim=config.latent_dim,
            output_size=self.output_size,
        )
        params = net.init(
            key, 
            jnp.ones(config.input_size), 
            jnp.ones(config.conditional_dim), 
            subkey
        )['params']

        # Set up the optimizer
        if getattr(config, 'weight_decay', 0.0) > 0:
            tx = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
        else:
            tx = optax.adam(config.learning_rate)

        # Create the state
        self.state = TrainState.create(apply_fn = net.apply, params = params, tx = tx)

    def preprocess_data(self, data: DataLoader, conversion: callable) -> None:
        """
        Preprocesses the training and validation data.
        Returns rescaled training and validation data arrays 
        as well as the scaler objects.
        For the CVAE, we scale the 2D flux arrays down to the shape determined through
        ``image_size`` and standardize them.

        Args:
            data (DataLoader): Data file with training and validation data.
            conversion (callable): Special conversion function to generate 
                                   parameter combinations that assist the training.

        Raises:
            ValueError: If ``nan``s are introduced when rescaling the flux densities.
        """

        X_scaler = ParameterScaler(
            scaler=StandardScalerJax(),
            parameter_names=data.parameter_names,
            conversion=conversion
        )

        y_scaler = DataScaler([
            ImageScaler(downscale=self.image_size,
                        upscale=(data.n_nus, data.n_times)),
            StandardScalerJax()
        ])

        train_X, val_X, X_scaler = data.preprocess_parameters(X_scaler)

        # first just the ImageScaler
        train_y, val_y, _ = data.preprocess_fluxes(y_scaler.scalers[0])
        train_y = train_y.reshape(-1, self.output_size)
        val_y = val_y.reshape(-1, self.output_size)

        # then standardize the down sampled fluxes
        train_y = y_scaler.scalers[1].fit_transform(train_y)
        val_y = y_scaler.scalers[1].transform(val_y)

        data._check_array_for_garbage(train_y, "train after standardization")
        data._check_array_for_garbage(val_y, "val after standardization")

        logger.info("Preprocessing data . . . done")

        return train_X, train_y, val_X, val_y, X_scaler, y_scaler
    
    @staticmethod
    @jax.jit
    def train_step(
        state: TrainState, 
        train_X: Float[Array, "n_batch_train ndim_input"], 
        train_y: Float[Array, "n_batch_train ndim_output"],
        rng: jax.random.PRNGKey,
        val_X: Float[Array, "n_batch_val ndim_output"] = None, 
        val_y: Float[Array, "n_batch_val ndim_output"] = None, 
    ) -> tuple[TrainState, Float[Array, "n_batch_train"], Float[Array, "n_batch_val"]]:

        # define a function that evaluates the NN
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
        best_state = state
        best_val_loss = jnp.inf

        start = time.time()

        for i in range(self.config.nb_epochs):
            # Do a single step
            rng, subkey = jax.random.split(rng)
            state, train_loss, val_loss, rng = self.train_step(state, train_X, train_y, subkey, val_X, val_y)
            # Save the losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            # Track the best model by validation loss
            if val_X is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = state
            # Report once in a while
            if i % self.config.nb_report == 0 and verbose:
                logger.info(f"Train loss at step {i+1}: {train_loss}")
                logger.info(f"Valid loss at step {i+1}: {val_loss}")
                logger.info(f"Best valid loss so far: {best_val_loss}")
                logger.info(f"Learning rate: {self.config.learning_rate}")
                logger.info("---")

        end = time.time()
        if verbose:
            logger.info(f"Training for {self.config.nb_epochs} took {end-start} seconds.")
            if val_X is not None:
                logger.info(f"Best validation loss: {best_val_loss}")

        self.trained_state = best_state if val_X is not None else state

        return self.trained_state, train_losses, val_losses
    
    @staticmethod
    def load_model(filename: str) -> tuple[TrainState, NeuralnetConfig]:
        """
        Load a model from a file.
    
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
        
        config = loaded_dict["config"]
        params = loaded_dict["params"]

        net = nn.CVAE(hidden_layer_sizes=config.hidden_layer_sizes, output_size= config.output_size)
        # Create train state without optimizer
        state = TrainState.create(apply_fn = net.apply, params = params, tx = optax.adam(config.learning_rate))

        return state, config