"""Store classes to load in trained models and give routines to let them generate lightcurves."""

# TODO: improve them with jax treemaps, since dicts are essentially pytrees

import os
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from functools import partial
from beartype import beartype as typechecker
from flax.training.train_state import TrainState
import pickle

import fiesta.train.neuralnets as fiesta_nn
from fiesta.utils import MinMaxScalerJax, inverse_svd_transform
from fiesta import models_utilities
from fiesta import utils

########################
### ABSTRACT CLASSES ###
########################

class LightcurveModel:
    """Abstract class for general light curve models"""
    
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
    
    def add_name(self, x: Array):
        return dict(zip(self.parameter_names, x))    
    
    def project_input(self, x: Array) -> dict[str, Array]:
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
    
    @partial(jax.jit, static_argnums=(0,)) # TODO: the jit here can come into conflict with scikit-learn methods used in project_output, maybe only jit self.comput_output
    def predict(self, x: dict[str, Array]) -> dict[str, Array]:
        """
        Generate the lightcurve y from the unnormalized and untransformed input x.
        Chains the projections with the actual computation of the output. E.g. if the model is a trained
        surrogate neural network, they represent the map from x tilde to y tilde. The mappings from
        x to x tilde and y to y tilde take care of projections (e.g. SVD projections) and normalizations.

        Args:
            x (dict[str, Array]): Input array, unnormalized and untransformed.

        Returns:
            Array: Output dict[str, Array], i.e., the desired raw light curve per filter
        """
        
        # Use saved parameter names to extract the parameters in the correct order into an array
        x_array = jnp.array([x[name] for name in self.parameter_names])
        x_tilde = self.project_input(x_array)
        y_tilde = self.compute_output(x_tilde)
        y = self.project_output(y_tilde)
        return y
    
    def __repr__(self) -> str:
        return self.name
    
class SurrogateLightcurveModel(LightcurveModel):
    """Abstract class for models that rely on a surrogate, in the form of a neural network."""
    
    directory: str
    metadata: dict
    X_scaler: MinMaxScalerJax
    y_scaler: dict[str, MinMaxScalerJax]
    models: dict[str, TrainState]
    times: Array
    tmin: Float
    tmax: Float
    parameter_names: list[str]
    
    def __init__(self,
                 name: str,
                 directory: str,
                 filters: list[str] = None,
                 times: Array = None) -> None:
        """_summary_

        Args:
            name (str): Name of the model
            directory (str): Directory with trained model states and projection metadata such as scalers.
            filters (list[str]): List of all the filters for which the model should be loaded.
        """
        super().__init__(name)
        self.directory = directory
        self.models = {}
        
        # Load the metadata for projections etc
        self.load_metadata()
        self.load_filters(filters)
        self.load_scalers()
        self.load_times(times)
        self.load_parameter_names()
        self.load_networks()
        
    def load_metadata(self) -> None:
        self.metadata_filename = os.path.join(self.directory, f"{self.name}_metadata.pkl")
        assert os.path.exists(self.metadata_filename), f"Metadata file {self.metadata_filename} not found - check the directory {self.directory}"
        meta_file = open(self.metadata_filename, "rb")
        self.metadata = pickle.load(meta_file)
        meta_file.close()
        
    def load_filters(self, filters: list[str] = None) -> None:
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
        self.filters = filters
        print(f"Loaded SurrogateLightcurveModel with filters {filters}")
        
    def load_scalers(self):
        self.X_scaler, self.y_scaler = {}, {}
        for filt in self.filters: 
            self.X_scaler[filt] = MinMaxScalerJax(min_val=self.metadata[filt]["X_scaler_min"], max_val=self.metadata[filt]["X_scaler_max"])
            self.y_scaler[filt] = self.metadata[filt]["y_scaler"]

            
    def load_times(self, times: Array = None) -> None:
        if times is None:
            times = jnp.array(self.metadata["times"])
        if times.min()<self.metadata["times"].min() or times.max()>self.metadata["times"].max():
            times = jnp.array(self.metadata["times"])
        self.times = times
        self.tmin = jnp.min(times)
        self.tmax = jnp.max(times)
        
    def load_networks(self) -> None:
        self.models = {}
        for filter in self.filters:
            filename = os.path.join(self.directory, f"{filter}.pkl")
            state, _ = fiesta_nn.load_model(filename)
            self.models[filter] = state
        
    def load_parameter_names(self) -> None:
        """Implement in child classes"""
        raise NotImplementedError
    
    def project_input(self, x: Array) -> dict[str, Array]:
        """
        Project the given input to whatever preprocessed input space we are in.

        Args:
            x (dict[str, Array]): Original input array

        Returns:
            dict[str, Array]: Transformed input array
        """
        x_tilde = {filter: self.X_scaler[filter].transform(x) for filter in self.filters}
        return x_tilde
    
    def compute_output(self, x: dict[str, Array]) -> dict[str, Array]:
        """
        Apply the trained flax neural network on the given input x.

        Args:
            x (dict[str, Array]): Input array of parameters per filter

        Returns:
            dict[str, Array]: _description_
        """
        # TODO: too convoluted, simplify
        return {filter: self.models[filter].apply_fn({'params': self.models[filter].params}, x[filter]) for filter in self.filters}
        
    def project_output(self, y: dict[str, Array]) -> dict[str, Array]:
        """
        Project the computed output to whatever preprocessed output space we are in.

        Args:
            y (dict[str, Array]): Output array

        Returns:
            dict[str, Array]: Output array transformed to the preprocessed space.
        """
        return {filter: self.y_scaler[filter].inverse_transform(y[filter]) for filter in self.filters}   
    
class SVDSurrogateLightcurveModel(SurrogateLightcurveModel):
    
    VA: dict[str, Array]
    svd_ncoeff: int
    
    def __init__(self, 
                 name: str, 
                 directory: str,
                 filters: list[str] = None,
                 times: Array = None):
        """
        Initialize a class to generate lightcurves from a Bulla trained model.
        
        """
        super().__init__(name=name, directory=directory, times=times, filters=filters)
        
        self.VA = {filt: self.metadata[filt]["VA"] for filt in filters}
        self.svd_ncoeff = {filt: self.metadata[filt]["svd_ncoeff"] for filt in filters}
        
    def load_parameter_names(self) -> None:
        raise NotImplementedError
            
    def project_output(self, y: dict[str, Array]) -> dict[str, Array]:
        """
        Apply the trained flax neural network on the given input x.

        Args:
            x (dict[str, Array]): Input array of parameters

        Returns:
            dict[str, Array]: _description_
        """
        output = {filter: inverse_svd_transform(y[filter], self.VA[filter], self.svd_ncoeff[filter]) for filter in self.filters}
        return super().project_output(output)
       
class BullaLightcurveModel(SVDSurrogateLightcurveModel):
    
    def __init__(self, 
                 name: str, 
                 directory: str,
                 filters: list[str] = None,
                 times: Array = None):
        
        super().__init__(name=name, directory=directory, filters=filters, times=times)
        
    def load_parameter_names(self) -> None:
        self.parameter_names = models_utilities.BULLA_PARAMETER_NAMES[self.name]
    
class AfterglowpyLightcurvemodel(SVDSurrogateLightcurveModel):
    
    def __init__(self,
                 name: str,
                 directory: str,
                 filters: list[str] = None,
                 times: Array = None):
        super().__init__(name=name, directory=directory, filters=filters, times=times)
        
    def load_parameter_names(self) -> None:
        self.parameter_names = self.metadata["parameter_names"]

class PCALightcurveModel(SurrogateLightcurveModel):

    def __init__(self,
                     name: str,
                     directory: str,
                     filters: list[str] = None, 
                     times: Array = None):
            
        super().__init__(name = name, directory= directory, filters = filters, times = times)

    def load_filters(self, filters: list[str] = None) -> None:
        self.nus = self.metadata['nus']
        self.Filters = []
        for filter in filters:
            try:
                Filter = utils.Filter(filter)
                if Filter.nu<self.nus[0] or Filter.nu>self.nus[-1]:
                    continue
                self.Filters.append(Filter)
            except:
                raise Exception(f"Filter {filter} not available.")
        
        self.filters = [filt.name for filt in self.Filters]
        if len(self.filters) == 0:
            raise ValueError(f"No filters found that match the trained frequency range {self.nus[0]:.3e} Hz to {self.nus[-1]:.3e} Hz.")

        print(f"Loaded SurrogateLightcurveModel with filters {self.filters}.")
            
    def load_scalers(self):
        self.X_scaler = self.metadata["X_scaler"]
        self.y_scaler = self.metadata["y_scaler"]
        #self.pca = self.metadata["pca"]

    def load_networks(self) -> None:
        filename = os.path.join(self.directory, f"{self.name}.pkl")
        state, _ = fiesta_nn.load_model(filename)
        self.models = state
    

    def project_input(self, x: Array) -> Array:
        """
        Project the given input to whatever preprocessed input space we are in.

        Args:
            x (Array): Original input array

        Returns:
            Array: Transformed input array
        """
        x_tilde = self.X_scaler.transform(x)
        return x_tilde
    
    def compute_output(self, x: Array) -> Array:
        """
        Apply the trained flax neural network on the given input x.

        Args:
            x (dict[str, Array]): Input array of parameters per filter

        Returns:
            dict[str, Array]: _description_
        """
        output = self.models.apply_fn({'params': self.models.params}, x)
        return output
        
    def project_output(self, y: Array) -> dict[str, Array]:
        """
        Project the computed output to whatever preprocessed output space we are in.

        Args:
            y (dict[str, Array]): Output array

        Returns:
            dict[str, Array]: Output array transformed to the preprocessed space.
        """
        #y = self.pca.inverse_transform(y)
        y = self.y_scaler.inverse_transform(y)

        y = jnp.reshape(y, (len(self.metadata["nus"]), len(self.times)))
        y = jnp.exp(y)
        
        def compute_mag_single_filter(nu):
            # TODO: get a check here that the filt.nu is in range of the meta data
            lambda_interp = lambda column: jnp.interp(nu, self.metadata["nus"], column) 
            mJys = jax.vmap(lambda_interp)(y.T)
            mag = -48.6 + -1 * jnp.log10(mJys) * 2.5 + -1 * (-26) * 2.5
            return mag
        
        filter_nus = jnp.array([filt.nu for filt in self.Filters])
        output_array = jax.vmap(compute_mag_single_filter)(filter_nus)
        output = dict(zip(self.filters, output_array))
        
        return output

    
    def predict_log_flux(self, x: Array) -> Array:
        """
        Predict the total log flux array for the parameters x.

        Args:
            x [Array]: raw parameter array

        Returns:
            log_flux [Array]: Array of log-fluxes.
        """

        x_tilde = self.X_scaler.transform(x)
        y = self.models.apply_fn({'params': self.models.params}, x_tilde)

        logflux = self.y_scaler.inverse_transform(y)
        return logflux

class AfterglowpyPCA(PCALightcurveModel):
    
    def __init__(self,
                 name: str,
                 directory: str,
                 filters: list[str] = None,
                 times: Array = None):
        super().__init__(name=name, directory=directory, filters=filters, times=times)
        
    def load_parameter_names(self) -> None:
        self.parameter_names = self.metadata["parameter_names"]