"""Store classes to load in trained models and give routines to let them generate lightcurves."""

# TODO: improve them with jax treemaps, since dicts are essentially pytrees

import os
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from functools import partial
from flax.training.train_state import TrainState
import pickle

import fiesta.train.neuralnets as fiesta_nn
from fiesta.conversions import mag_app_from_mag_abs, apply_redshift
from fiesta import utils


########################
### ABSTRACT CLASSES ###
########################

class SurrogateModel:
    """Abstract class for general surrogate models"""
    
    name: str
    directory: str 
    filters: list[str]
    parameter_names: list[str]
    times: Array
    
    def __init__(self, 
                 name: str,
                 directory: str) -> None:
        self.name = name
        self.directory = directory

        self.load_metadata()
        
        self.filters = []
    
    def add_name(self, x: Array):
        return dict(zip(self.parameter_names, x))
    
    def load_metadata(self) -> None:
        print(f"Loading metadata for model {self.name}.")
        self.metadata_filename = os.path.join(self.directory, f"{self.name}_metadata.pkl")
        assert os.path.exists(self.metadata_filename), f"Metadata file {self.metadata_filename} not found - check the directory {self.directory}"
        
        # open the file
        with open(self.metadata_filename, "rb") as meta_file:
            self.metadata = pickle.load(meta_file)
        
        # make the scaler objects attributes
        self.X_scaler = self.metadata["X_scaler"]
        self.y_scaler = self.metadata["y_scaler"]

        # load parameter names
        self.parameter_names = self.metadata["parameter_names"]
        print(f"This surrogate {self.name} should only be used in the following parameter ranges:")
        from ast import literal_eval
        parameter_distributions = literal_eval(self.metadata["parameter_distributions"])
        for key in parameter_distributions.keys():
            print(f"\t {key}: {parameter_distributions[key][:2]}")

        #load times
        self.times = self.metadata["times"]

        #load nus
        if "nus" in self.metadata.keys():
            self.nus = self.metadata["nus"]
    
    
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
    
    def convert_to_mag(self, y: Array, x: dict[str, Array]) -> tuple[Array, dict[str, Array]]:
        raise NotImplementedError
    
    @partial(jax.jit, static_argnums=(0,))
    def predict(self, x: dict[str, Array]) -> tuple[Array, dict[str, Array]]:
        """
        Generate the apparent magnitudes from the unnormalized and untransformed input x.
        Chains the projections with the actual computation of the output. E.g. if the model is a trained
        surrogate neural network, they represent the map from x tilde to y tilde. The mappings from
        x to x tilde and y to y tilde take care of projections (e.g. SVD projections) and normalizations.

        Args:
            x (dict[str, Array]): Input array, unnormalized and untransformed.

        Returns:
            times
            mag (dict[str, Array]): The desired magnitudes per filter
        """
        
        # Use saved parameter names to extract the parameters in the correct order into an array
        x_array = jnp.array([x[name] for name in self.parameter_names])

        # apply the NN
        x_tilde = self.project_input(x_array)
        y_tilde = self.compute_output(x_tilde)
        y = self.project_output(y_tilde)

        # convert the NN output to apparent magnitude
        times, mag = self.convert_to_mag(y, x)

        return times, mag
    
    def predict_abs_mag(self, x: dict[str, Array]) -> tuple[Array, dict[str, Array]]:
        x["luminosity_distance"] = 1e-5
        x["redshift"] = 0.

        return self.predict(x)
    
    def vpredict(self, X: dict[str, Array]) -> tuple[Array, dict[str, Array]]:
        """
        Vectorized prediction function to calculate the apparent magnitudes for several inputs x at the same time.
        """
        
        X_array = jnp.array([X[name] for name in X.keys()]).T

        def predict_single(x):
            param_dict = {key: x[j] for j, key in enumerate(X.keys())}
            return self.predict(param_dict)
        
        times, mag_apps = jax.vmap(predict_single)(X_array)

        return times[0], mag_apps
    
    def __repr__(self) -> str:
        return self.name
    
class LightcurveModel(SurrogateModel):
    """Class of surrogate models that predicts the magnitudes per filter."""
    
    directory: str
    metadata: dict
    X_scaler: object
    y_scaler: dict[str, object]
    models: dict[str, TrainState]
    
    def __init__(self,
                 name: str,
                 directory: str,
                 filters: list[str] = None) -> None:
        """_summary_

        Args:
            name (str): Name of the model
            directory (str): Directory with trained model states and projection metadata such as scalers.
            filters (list[str]): List of all the filters for which the model should be loaded.
        """
        super().__init__(name, directory)
        
        # Load the filters and networks
        self.load_filters(filters)
        self.load_networks()
        
    def load_filters(self, filters_args: list[str] = None) -> None:
        # Save those filters that were given and that were trained and store here already
        pkl_files = [file for file in os.listdir(self.directory) if file.endswith(".pkl") or file.endswith(".pickle")]
        all_available_filters = [(file.split(".")[0]).split("_")[1] for file in pkl_files]
        
        if filters_args is None:
            # Use all filters that the surrogate model supports
            filters = all_available_filters
        else:
            # Fetch those filters specified by the user that are available
            filters = [f for f in filters_args if f in all_available_filters]
        
        if len(filters) == 0:
            raise ValueError(f"No filters found in {self.directory} that match the given filters {filters_args}.")
        self.filters = filters
        self.Filters = [filters.Filter(filt) for filt in self.filters]
        print(f"Loaded SurrogateLightcurveModel with filters {self.filters}.")
        
    def load_networks(self) -> None:
        self.models = {}
        for filter in self.filters:
            filename = os.path.join(self.directory, f"{self.name}_{filter}.pkl")
            state, _ = fiesta_nn.MLP.load_model(filename)
            self.models[filter] = state
    
    def project_input(self, x: Array) -> Array:
        """
        Project the given input to whatever preprocessed input space we are in.

        Args:
            x (dict[str, Array]): Original input array

        Returns:
            dict[str, Array]: Transformed input array
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
        def apply_model(filter):
            model = self.models[filter]
            output = model.apply_fn({'params': model.params}, x)
            return output
        
        y = jax.tree.map(apply_model, self.filters) # avoid for loop with jax.tree.map 
        return dict(zip(self.filters, y))
        
    def project_output(self, y: dict[str, Array]) -> dict[str, Array]:
        """
        Project the computed output to whatever preprocessed output space we are in.

        Args:
            y (dict[str, Array]): Output array

        Returns:
            dict[str, Array]: Output array transformed to the preprocessed space.
        """
        def inverse_transform(filter):
            y_scaler = self.y_scaler[filter]
            output = y_scaler.inverse_transform(y[filter])
            return output
        
        y = jax.tree.map(inverse_transform, self.filters) # avoid for loop with jax.tree.map
        return jnp.array(y)
    
    def convert_to_mag(self, y: Array, x: dict[str, Array]) -> tuple[Array, dict[str, Array]]:
        mag_abs = y
        mag_app = mag_app_from_mag_abs(mag_abs, x["luminosity_distance"])
        return self.times, dict(zip(self.filters, mag_app))

class FluxModel(SurrogateModel):
    """Class of surrogate models that predicts the 2D spectral flux density array."""

    def __init__(self,
                 name: str,
                 directory: str,
                 filters: list[str] = None, 
                 model_type: str = "MLP"):
        self.model_type = model_type # TODO: make this switch nicer somehow maybe
        super().__init__(name, directory)

        # Load the filters and networks
        self.load_filters(filters)
        self.load_networks()

    def load_filters(self, filters: list[str] = None) -> None:
        self.nus = self.metadata['nus']
        self.Filters = []
        for filter in filters:
            try:
                Filter = filters.Filter(filter)
                if Filter.nu<self.nus[0] or Filter.nu>self.nus[-1]:
                    continue
                self.Filters.append(Filter)
            except:
                raise Exception(f"Filter {filter} not available.")
        
        self.filters = [filt.name for filt in self.Filters]
        if len(self.filters) == 0:
            raise ValueError(f"No filters found that match the trained frequency range {self.nus[0]:.3e} Hz to {self.nus[-1]:.3e} Hz.")

        print(f"Loaded SurrogateLightcurveModel with filters {self.filters}.")

    def load_networks(self) -> None:
        filename = os.path.join(self.directory, f"{self.name}.pkl")
        if self.model_type == "MLP":
            state, _ = fiesta_nn.MLP.load_model(filename)
            latent_dim = 0
        elif self.model_type == "CVAE":
           state, _ = fiesta_nn.CVAE.load_model(filename)
           latent_dim = state.params["layers_0"]["kernel"].shape[0] - len(self.parameter_names)
        else:
            raise ValueError(f"Model type must be either 'MLP' or 'CVAE'.")
        self.latent_vector = jnp.array(jnp.zeros(latent_dim)) # TODO: how to get latent vector?
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
        x = jnp.concatenate((self.latent_vector, x))
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
        y = self.y_scaler.inverse_transform(y)
        y = jnp.reshape(y, (len(self.nus), len(self.times)))
        
        return y
    
    def convert_to_mag(self, y: Array, x: dict[str, Array]) -> tuple[Array, dict[str, Array]]:

        mJys = jnp.exp(y)

        mJys_obs, times_obs, nus_obs = apply_redshift(mJys, self.times, self.nus, x["redshift"])
        # TODO: Add EBL table here at some point

        mag_abs = jax.tree.map(lambda Filter: Filter.get_mag(mJys_obs, nus_obs), 
                               self.Filters)
        mag_abs = jnp.array(mag_abs)
        
        mag_app = mag_app_from_mag_abs(mag_abs, x["luminosity_distance"])
        
        return times_obs, dict(zip(self.filters, mag_app))
    
    def predict_log_flux(self, x: Array) -> Array:
        """
        Predict the total log flux array for the parameters x.

        Args:
            x [Array]: raw parameter array

        Returns:
            log_flux [Array]: Array of log-fluxes.
        """
        x_tilde = self.X_scaler.transform(x)
        x_tilde = jnp.concatenate((self.latent_vector, x_tilde))
        y = self.models.apply_fn({'params': self.models.params}, x_tilde)

        logflux = self.y_scaler.inverse_transform(y)
        logflux = logflux.reshape(len(self.nus), len(self.times))
        return logflux


#################
# MODEL CLASSES #
#################

class BullaLightcurveModel(LightcurveModel):
    
    def __init__(self, 
                 name: str, 
                 directory: str,
                 filters: list[str] = None):
        
        super().__init__(name=name, directory=directory, filters=filters)

class AfterglowFlux(FluxModel):
    
    def __init__(self,
                 name: str,
                 directory: str,
                 filters: list[str] = None,
                 model_type: str = "MLP"):
        super().__init__(name=name, directory=directory, filters=filters, model_type=model_type)
    