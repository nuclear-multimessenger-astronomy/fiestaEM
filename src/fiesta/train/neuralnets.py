import time

import numpy as np

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

import flax
from flax.training.train_state import TrainState
from ml_collections import ConfigDict
import optax
import pickle

from fiesta.logging import logger
from fiesta.train import DataLoader
import fiesta.train.nn_architectures as nn
from fiesta.scalers import (
    ParameterScaler,
    DataScaler,
    StandardScalerJax,
    PCADecomposer,
    ImageScaler
)

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

################
### TRAINING ###
################

class NN:
    """
    Abstract base class for the NN architecture wrappers of flax networks below.
    """

    def train_loop(
        self,       
        train_X: Float[Array, "n_batch_train ndim_input"], 
        train_y: Float[Array, "n_batch_train ndim_output"],
        val_X: Float[Array, "n_batch_val ndim_output"] = None, 
        val_y: Float[Array, "n_batch_val ndim_output"] = None,
        verbose: bool = True
    ) -> tuple[TrainState, Array, Array]:
        raise NotImplementedError

    def save_model(self, outfile: str) -> None:
        """
        Serialize and save the model to a file.

        Raises:
            ValueError: If the provided file extension is not .pkl or .pickle.

        Args:
            outfile (str): The pickle file to which we save the serialized model.
        """

        if not outfile.endswith(".pkl") and not outfile.endswith(".pickle"):
            raise ValueError("For now, only .pkl or .pickle extensions are supported.")

        serialized_dict = serialize(self.trained_state, self.config)
        with open(outfile, 'wb') as handle:
            pickle.dump(serialized_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
        
        config: NeuralnetConfig = loaded_dict["config"]
        params = loaded_dict["params"]

        net = nn.CVAE(hidden_layer_sizes=config.hidden_layer_sizes, output_size= config.output_size)
        # Create train state without optimizer
        state = TrainState.create(apply_fn = net.apply, params = params, tx = optax.adam(config.learning_rate))

        return state, config
        

class MLP(NN):
    """
    Classical multi-layer perceptron using the flax-interface.

    Args: 
        config (NeurnetConfig): NN config dictionary. Its ``output_size``
                                will determine to the number of PCA components kept
                                after data preprocessing.
        key (PRNGKey, optional): Random key for initialization. Defaults to ``21``.
    """
    def __init__(
            self,
            config: NeuralnetConfig,
            key: jax.random.PRNGKey = jax.random.key(21)
        ) -> None:

        self.config = config
        dropout_rate = getattr(config, 'dropout_rate', 0.0)

        # Create the neural network
        net = nn.MLP(layer_sizes=config.layer_sizes, dropout_rate=dropout_rate)
        params = net.init(key, jnp.ones(config.input_size), train=False)['params']

        # Set up the optimizer
        if getattr(config, 'weight_decay', 0.0) > 0:
            tx = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
        else:
            tx = optax.adam(config.learning_rate)

        # Create the state
        self.state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    def preprocess_data(self, data: DataLoader, conversion: callable) -> None:
        """
        Preprocesses the training and validation data.
        Returns rescaled training and validation data arrays 
        as well as the scaler objects.
        For the MLP, we perform PCA decomposition and keep the number of 
        PCA components specified through the ``output_size`` of the NN.

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
            PCADecomposer(n_components=self.config.output_size)
        ])

        (   
            train_X, 
            train_y, 
            val_X, 
            val_y, 
            X_scaler, 
            y_scaler
                    ) = data.preprocess_data(X_scaler, y_scaler) 

        if jnp.any(jnp.isnan(train_y)) or jnp.any(jnp.isnan(val_y)):
            raise ValueError(f"Data preprocessing introduced nans."
                              "Check raw data for nans of infs or "
                              "vanishing variance in a specific entry.")

        logger.info("PCA decomposition accounts for "
                    f"{jnp.sum(y_scaler.scalers[0].explained_variance_ratio_).item() *100 :.2f} %"
                    " of the total variance in the training data. This value is hopefully close to 1.")
        logger.info("Preprocessing data . . . done")

        return train_X, train_y, val_X, val_y, X_scaler, y_scaler

    @staticmethod
    @jax.jit
    def train_step(state, batch_X, batch_y, dropout_rng, component_weights):
        def loss_fn(params):
            pred_y = state.apply_fn({'params': params}, batch_X, train=True,
                                    rngs={'dropout': dropout_rng})
            per_sample = jax.vmap(
                lambda y, p: jnp.sum(component_weights * (y - p) ** 2)
            )(batch_y, pred_y)
            return jnp.mean(per_sample)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @staticmethod
    @jax.jit
    def eval_step(state, X, y, component_weights):
        pred_y = state.apply_fn({'params': state.params}, X, train=False)
        per_sample = jax.vmap(
            lambda y, p: jnp.sum(component_weights * (y - p) ** 2)
        )(y, pred_y)
        return jnp.mean(per_sample)

    def train_loop(self,
                   train_X: Float[Array, "n_batch_train ndim_input"],
                   train_y: Float[Array, "n_batch_train ndim_output"],
                   val_X: Float[Array, "n_batch_val ndim_output"] = None,
                   val_y: Float[Array, "n_batch_val ndim_output"] = None,
                   verbose: bool = True):

        total_steps = self.config.nb_epochs

        # Component weights for smoothness regularization
        n_pca = train_y.shape[1]
        sw = getattr(self.config, 'pca_smoothness_weight', 0.0)
        ss = getattr(self.config, 'pca_smoothness_start', 0)
        if sw > 0:
            decay = jnp.exp(sw * jnp.clip(jnp.arange(n_pca) - ss, 0, None) / n_pca)
            component_weights = jnp.ones(n_pca).at[ss:].set(decay[ss:])
        else:
            component_weights = jnp.ones(n_pca)

        # Optionally rebuild optimizer with cosine LR schedule
        if getattr(self.config, 'use_cosine_schedule', False):
            schedule_fn = optax.cosine_decay_schedule(
                init_value=self.config.learning_rate,
                decay_steps=total_steps,
                alpha=getattr(self.config, 'cosine_alpha', 0.01))
            parts = []
            if getattr(self.config, 'max_grad_norm', 0.0) > 0:
                parts.append(optax.clip_by_global_norm(self.config.max_grad_norm))
            wd = getattr(self.config, 'weight_decay', 0.0)
            if wd > 0:
                parts.append(optax.adamw(schedule_fn, weight_decay=wd))
            else:
                parts.append(optax.adam(schedule_fn))
            tx = optax.chain(*parts) if len(parts) > 1 else parts[0]
            self.state = TrainState.create(
                apply_fn=self.state.apply_fn,
                params=self.state.params,
                tx=tx)

        train_losses, val_losses = [], []
        state = self.state
        best_state = state
        best_val_loss = jnp.inf
        rng = jax.random.key(2025)

        start = time.time()

        for i in range(self.config.nb_epochs):
            rng, dropout_rng = jax.random.split(rng)
            state, epoch_loss = self.train_step(
                state, train_X, train_y, dropout_rng, component_weights)

            # Evaluate on full validation set
            if val_X is not None:
                val_loss = self.eval_step(state, val_X, val_y, component_weights)
            else:
                val_loss = jnp.zeros_like(epoch_loss)

            train_losses.append(epoch_loss)
            val_losses.append(val_loss)

            # Track the best model by validation loss
            if val_X is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = state

            # Report once in a while
            if i % self.config.nb_report == 0 and verbose:
                logger.info(f"Train loss at step {i+1}: {epoch_loss}")
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

        dropout_rate = getattr(config, 'dropout_rate', 0.0)
        net = nn.MLP(config.layer_sizes, dropout_rate=dropout_rate)
        # Create train state without optimizer
        state = TrainState.create(apply_fn=net.apply, params=params, tx=optax.adam(config.learning_rate))

        return state, config            