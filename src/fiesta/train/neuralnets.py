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
                 weight_decay: Float = 0.0,
                 batch_size: int = 128,
                 nb_epochs: Int = 1_000,
                 nb_report: Int = None,
                 dropout_rate: float = 0.0,
                 use_cosine_schedule: bool = False,
                 cosine_alpha: float = 0.01,
                 max_grad_norm: float = 0.0,
                 pca_smoothness_weight: float = 0.0,
                 pca_smoothness_start: int = 0):

        super().__init__()
        self.name = name
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layer_sizes = [*hidden_layer_sizes, output_size]
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        if nb_report is None:
            nb_report = self.nb_epochs // 10
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


class CVAE:
    def __init__(self,
                 config: NeuralnetConfig,
                 conditional_dim: Int,
                 key: jax.random.PRNGKey = jax.random.key(21)):
        self.config = config
        net = nn.CVAE(hidden_layer_sizes=config.hidden_layer_sizes, latent_dim=config.latent_dim, output_size=config.output_size)
        key, subkey, subkey2 = jax.random.split(key, 3)

        params = net.init(subkey, jnp.ones(config.output_size), jnp.ones(conditional_dim), subkey2)['params']
        if getattr(config, 'weight_decay', 0.0) > 0:
            tx = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
        else:
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
        dropout_rate = getattr(config, 'dropout_rate', 0.0)
        net = nn.MLP(layer_sizes=config.layer_sizes, dropout_rate=dropout_rate)
        key, subkey = jax.random.split(key)
        params = net.init(subkey, jnp.ones(input_ndim), train=False)['params']
        if getattr(config, 'weight_decay', 0.0) > 0:
            tx = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
        else:
            tx = optax.adam(config.learning_rate)
        self.state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

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

        n_train = train_X.shape[0]
        batch_size = min(self.config.batch_size, n_train)
        n_batches = max(1, n_train // batch_size)
        total_steps = self.config.nb_epochs * n_batches

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
            # Shuffle training data
            rng, shuffle_rng = jax.random.split(rng)
            perm = jax.random.permutation(shuffle_rng, n_train)
            X_shuf = train_X[perm]
            y_shuf = train_y[perm]

            # Mini-batch loop
            epoch_loss = 0.0
            for b in range(n_batches):
                s = b * batch_size
                bX = jax.lax.dynamic_slice(X_shuf, (s, 0),
                                           (batch_size, train_X.shape[1]))
                by = jax.lax.dynamic_slice(y_shuf, (s, 0),
                                           (batch_size, train_y.shape[1]))
                rng, dropout_rng = jax.random.split(rng)
                state, batch_loss = self.train_step(
                    state, bX, by, dropout_rng, component_weights)
                epoch_loss += batch_loss
            epoch_loss /= n_batches

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

        dropout_rate = getattr(config, 'dropout_rate', 0.0)
        net = nn.MLP(config.layer_sizes, dropout_rate=dropout_rate)
        # Create train state without optimizer
        state = TrainState.create(apply_fn=net.apply, params=params, tx=optax.adam(config.learning_rate))

        return state, config            