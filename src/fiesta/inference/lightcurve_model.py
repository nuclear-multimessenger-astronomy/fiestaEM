"""Store classes to load in trained models and give routines to let them generate lightcurves."""

# TODO: improve them with jax treemaps, since dicts are essentially pytrees

import os
import jax
import jax.numpy as jnp
from jaxtyping import Array
from functools import partial
from beartype import beartype as typechecker
from flax.training.train_state import TrainState
import joblib

import fiesta.train.neuralnets as fiesta_nn
from fiesta.utils import MinMaxScalerJax, inverse_svd_transform
from fiesta import models_utilities


class LightcurveModel:
    """Abstract class"""
    
    name: str 
    filters: list[str]
    parameter_names: list[str]
    times: Array
    
    def __init__(self, 
                 name: str) -> None:
        self.name = name
        self.filters = []
        self.parameter_names = []
        self.times = jnp.array([])
        
    def project_input(self, x: dict[str, Array]) -> dict[str, Array]:
        """
        Project the given input to whatever preprocessed input space we are in. 
        By default (i.e., in this base class), the projection is the identity function.

        Args:
            x (Array): Input array

        Returns:
            Array: Input array transformed to the preprocessed space.
        """
        return x
    
    def compute_output(self, x: dict[str, Array]) -> dict[str, Array]:
        """
        Compute the output (untransformed) from the given, transformed input. 
        This is the main method that needs to be implemented by subclasses.

        Args:
            x (Array): Input array

        Returns:
            Array: Output array
        """
        raise NotImplementedError
        
    def project_output(self, y: dict[str, Array]) -> dict[str, Array]:
        """
        Project the computed output to whatever preprocessed output space we are in. 
        By default (i.e., in this base class), the projection is the identity function.

        Args:
            y (Array): Output array

        Returns:
            Array: Output array transformed to the preprocessed space.
        """
        return y
    
    @partial(jax.jit, static_argnums=(0,))
    def predict(self, x: dict[str, Array]) -> dict[str, Array]:
        """
        Generate the lightcurve y from the unnormalized and untransformed input x.
        Chains the projections with the actual computation of the output. E.g. if the model is a trained
        surrogate neural network, they represent the map from x tilde to y tilde. The mappings from
        x to x tilde and y to y tilde take care of projections (e.g. SVD projections) and normalizations.

        Args:
            x (Array): Input array, unnormalized and untransformed.

        Returns:
            Array: Output dict[str, Array], i.e., the desired raw light curve per filter
        """
        x_tilde = self.project_input(jnp.array([x[name] for name in self.parameter_names]))
        y_tilde = self.compute_output(x_tilde)
        y = self.project_output(y_tilde)
        return y
    
    def __repr__(self) -> str:
        return self.name
    
class BullaLightcurveModel(LightcurveModel):
    
    directory: str
    X_scaler: MinMaxScalerJax
    y_scaler: dict[str, MinMaxScalerJax]
    VA: dict[str, Array]
    models: dict[str, TrainState]
    times: Array
    
    def __init__(self, 
                 name: str, 
                 directory: str,
                 times: Array = None,
                 filters: list[str] = None):
        """
        Initialize a class to generate lightcurves from a Bulla trained model.

        Args:
            name (str): Name of the model
            directory (str): Directory with trained model states and projection metadata such as scalers.
            filters (list[str]): List of all the filters for which the model should be loaded.
        """
        super().__init__(name=name)
        self.directory = directory
        
        # Save those filters that were given and that were trained and store here already
        pkl_files = [file for file in os.listdir(self.directory) if file.endswith(".pkl") or file.endswith(".pickle")]
        all_available_filters = [file.split(".")[0] for file in pkl_files]
        
        if filters is None:
            # Use all filters that the surrogate model supports
            filters = all_available_filters
        else:
            # Fetch those filters specified by the user that are available
            filters = [f.replace(":", "_") for f in filters]
            filters = [f for f in filters if f in all_available_filters]
        
        if len(filters) == 0:
            raise ValueError(f"No filters found in {self.directory} that match the given filters {filters}")
        print(f"Loaded BullaLightcurveModel with filters {filters}")
        self.filters = filters
        
        # TODO: this is a bit cumbersome... Is there a better way to do it?
        
        # Load the metadata for projections etc
        metadata = joblib.load(os.path.join(self.directory, f"{self.name}.joblib"))
        
        # TODO: check for time range and trained model time range
        if times is not None:
            times = jnp.array(metadata["times"])
            
        self.times = times
        
        min_val, max_val = metadata["X_scaler_min"], metadata["X_scaler_max"]
        self.X_scaler = MinMaxScalerJax(min_val=min_val, max_val=max_val)
        
        min_val, max_val = metadata["y_scaler_min"], metadata["y_scaler_max"]
        self.y_scaler = {}
        for filt in self.filters:
            self.y_scaler[filt] = MinMaxScalerJax(min_val=min_val[filt], max_val=max_val[filt])
        
        # TODO: do we have to explicitly convert to jnp.arrays?
        self.VA = metadata["VA"]
        self.nsvd_coeff = metadata["nsvd_coeff"]
        
        # Load the trained model states
        self.models = {}
        for filter in filters:
            filename = os.path.join(self.directory, f"{filter}.pkl")
            state, _ = fiesta_nn.load_model(filename)
            self.models[filter] = state
            
        # Also save the parameter names and times
        self.parameter_names = models_utilities.BULLA_PARAMETER_NAMES[name]
        self.times = jnp.array(metadata["times"])
        self.tmin = self.times[0]
        self.tmax = self.times[-1]
            
    def project_input(self, x: Array) -> Array:
        """
        Project the given input to whatever preprocessed input space we are in.

        Args:
            x (dict[str, Array]): Original input array

        Returns:
            dict[str, Array]: Transformed input array
        """
        x_tilde = {filter: self.X_scaler.transform(x) for filter in self.filters}
        return x_tilde
            
    def compute_output(self, x: dict[str, Array]) -> dict[str, Array]:
        """
        Apply the trained flax neural network on the given input x.

        Args:
            x (dict[str, Array]): Input array of parameters

        Returns:
            dict[str, Array]: _description_
        """
        
        # TODO: too convoluted, simplify
        return {filter: self.models[filter].apply_fn({'params': self.models[filter].params}, x[filter]) for filter in self.filters}

    def project_output(self, y: dict[str, Array]) -> dict[str, Array]:
        """
        Apply the trained flax neural network on the given input x.

        Args:
            x (dict[str, Array]): Input array of parameters

        Returns:
            dict[str, Array]: _description_
        """
        
        output = {filter: inverse_svd_transform(y[filter], self.VA[filter], self.nsvd_coeff) for filter in self.filters}
        output = {filter: self.y_scaler[filter].inverse_transform(output[filter]) for filter in self.filters}

        return output