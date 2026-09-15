"""Class for the multi-layer perceptron."""
import pickle
import time

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
    PCADecomposer,
)

from .base import NN
from .utils import NeuralnetConfig
        

class MLP(NN):
    """
    Classical multi-layer perceptron using the flax-interface.

    Args: 
        config (NeuralnetConfig): NN config dictionary. Its ``output_size``
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
        Load an MLP from file.

        Args:
            filename (str): Filename of the model to be loaded.
    
        Raises:
            ValueError: If there is something wrong with loading, since lots of things can go wrong here.
    
        Returns:
            tuple[TrainState, NeuralnetConfig]: The TrainState object loaded from the file and the NeuralnetConfig object.
        """
        with open(filename, 'rb') as handle:
            loaded_dict = pickle.load(handle)

        config = loaded_dict["config"]
        params = loaded_dict["params"]

        dropout_rate = getattr(config, 'dropout_rate', 0.0)
        net = nn.MLP(config.layer_sizes, dropout_rate=dropout_rate)
        # Create train state without optimizer
        state = TrainState.create(apply_fn=net.apply, params=params, tx=optax.adam(config.learning_rate))

        return state, config            