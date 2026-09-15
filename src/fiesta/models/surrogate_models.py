"""Store classes to load in trained machine-learning surrogates and give routines to let them generate lightcurves."""

from ast import literal_eval
from typing import Iterable
import dill
from functools import partial
import os
from pathlib import Path


import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from flax.training.train_state import TrainState

from fiesta.logging import logger
import fiesta.train.neuralnets as fiesta_nn
from fiesta.conversions import mag_app_from_mag_abs, apply_redshift
import fiesta.filters as fiesta_filters
from fiesta.surrogates import built_in_surrogates, download_surrogate

from .base import FiestaModel


def get_default_directory(name):
    
    for model_name, surrogate_dir, _ in built_in_surrogates():
        if name==model_name:
            if not os.path.exists(surrogate_dir / "model"):
                raise OSError(f"Could not find model directory for name {name} in {surrogate_dir}. Please change the name or provide a path manually.")
            return surrogate_dir / "model"
    
    logger.info(f"Could not find {name} in built-in surrogates. Attempting download.")
    download_ok, surrogate_dir = download_surrogate(name)
    if download_ok:
        return surrogate_dir / "model"
    else:
        raise ValueError(f"No model directory provided, but could not find built-in surrogate {name} or download it.")


########################
### ABSTRACT CLASSES ###
########################

class Surrogate(FiestaModel):
    """Abstract class for general surrogate models"""

    directory: str 
    
    def __init__(
            self, 
            name: str,
            filters: list[str] | list[fiesta_filters.Filter],
            directory: str | None =None
        ) -> None:

        super().__init__(name, filters)
        
        if directory is None:
            self.directory = get_default_directory(name)
        else:
            self.directory = directory

        self.load_metadata()
    
    def load_metadata(self) -> None:
        metadata_files = [f for f in os.listdir(self.directory) if f.endswith("_metadata.pkl")]

        if len(metadata_files)==0:
            raise OSError(f"Metadata file not found - check the directory {self.directory}.")
            
        if len(metadata_files)>1:
            raise OSError(f"Found multiple metadata files in directory {self.directory}. Remove the ones you don't wish to load from there.")
        
        metadata_filename = os.path.join(self.directory, metadata_files[0])
        
        # open the file
        with open(metadata_filename, "rb") as meta_file:
            metadata = dill.load(meta_file)
        
        # make the scaler objects attributes
        self.X_scaler = metadata["X_scaler"]
        self.y_scaler = metadata["y_scaler"]
        
        # check the model type
        self.model_type = metadata["model_type"]

        # load parameter names
        self.parameter_names = metadata["parameter_names"]
        self.parameter_distributions = literal_eval(metadata["parameter_distributions"])
        logger.info(f"Loading surrogate {self.name}. This surrogate should only be used in the following parameter ranges:")
        for key in self.parameter_distributions.keys():
            logger.info(f"\t {key}: {self.parameter_distributions[key][:2]}")

        #load times
        self.times = metadata["times"]
        logger.info(f"Surrogate {self.name} is loading with source-frame time range {self.times[[0, -1]]} days.")

        #load nus
        if "nus" in metadata.keys():
            self.nus = metadata["nus"]
    
    
    def project_input(self, x: Array) -> dict[str, Array]:
        raise NotImplementedError
    
    def compute_output(self, x: dict[str, Array]) -> dict[str, Array]:
        raise NotImplementedError
        
    def project_output(self, y: dict[str, Array]) -> dict[str, Array]:
        return NotImplementedError
    
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
            tuple:
                times (Array): time array in observer frame
                mag (dict[str, Array]): The predicted magnitudes per filter
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
    
    def _on_filters_changed(self) -> None:
        jax.clear_caches()


############################
### Actual surrogate API ###
############################

class FluxSurrogate(Surrogate):
    """
    Class of surrogate models that predicts the 2D spectral flux density array.
    
    Args:
        name (str): Name of the model
        filters (list[str]): List of all the filters for which the model should be loaded.
        directory (str): Directory with trained model states and projection metadata such as scalers. Defaults to None, in which case there will be an attempt to load from the repo based on name.

    """

    nus: Array

    def __init__(
            self,
            name: str,
            filters: list[str],
            directory: str = None
        ) -> None:
        super().__init__(name, filters, directory)

        # Load the filters and networks
        self.load_filters(filters)
        self.load_networks()
        logger.info(f"Loaded for surrogate {self.name} from {self.directory}.")

    def load_filters(self, filters: list[str] = None) -> None:
        self.Filters = []
        for filter in filters:
            try:
                Filter = fiesta_filters.Filter(filter)
            except:
                raise Exception(f"Filter {filter} not available.")                
                        
            if Filter.nus[0]<self.nus[0] or Filter.nus[-1]>self.nus[-1]:
                    logger.warning(f"Filter {filter} outside of frequency range of {self.name} surrogate. Removing from model filters.")
            else: 
                self.Filters.append(Filter)
        
        self.filters = [filt.name for filt in self.Filters]
        if len(self.filters) == 0:
            raise ValueError(f"No filters found that match the trained frequency range {self.nus[0]:.3e} Hz to {self.nus[-1]:.3e} Hz.")

        logger.info(f"Surrogate {self.name} is loading with the following filters: {self.filters}.")

    def load_networks(self) -> None:
        filename = [f for f in os.listdir(self.directory) if (f.endswith(".pkl") and "metadata" not in f)][0]
        filename = os.path.join(self.directory, filename)

        if self.model_type == "MLP":
            state, _ = fiesta_nn.MLP.load_model(filename)
            latent_dim = 0
        elif self.model_type == "CVAE":
           state, _ = fiesta_nn.CVAE.load_model(filename)
           x_tilde_dim = self.X_scaler.transform(jnp.zeros(len(self.parameter_names)).reshape(1, -1) ).shape[1]
           latent_dim = state.params["layers_0"]["kernel"].shape[0] - x_tilde_dim
        else:
            raise ValueError(f"Model type must be either 'MLP' or 'CVAE'.")
        self.latent_vector = jnp.array(jnp.zeros(latent_dim))
        self.models = state
    
    def project_input(self, x: Array) -> Array:
        """
        Project the given input to whatever preprocessed input space we are in.

        Args:
            x (Array): Original input array

        Returns:
            Array: Transformed input array
        """
        x = x.reshape(1,-1)
        x_tilde = self.X_scaler.transform(x)
        x_tilde = x_tilde.reshape(-1)
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

        mJys = jnp.power(10, y)

        times_obs, nus_obs, mJys_obs= apply_redshift(mJys, self.times, self.nus, x["redshift"])
        # TODO: Add EBL table here at some point
        
        # shift the times according to the timeshift parameter if it is present in the input dictionary
        if "timeshift" in x.keys():
            times_obs = times_obs + x["timeshift"]

        mag_abs = jax.tree.map(lambda Filter: Filter.get_mag(mJys_obs, nus_obs), 
                               self.Filters)
        mag_abs = jnp.array(mag_abs)
        
        mag_app = mag_app_from_mag_abs(mag_abs, x["luminosity_distance"])
        
        return times_obs, dict(zip(self.filters, mag_app))
    
    def predict_log_flux(self, x: dict[str, Array]) -> Array:
        """
        Predict the total log10 flux array for the parameters x.

        Args:
            x (dict[str, Array]): Input parameters, unnormalized and untransformed.

        Returns:
            tuple:
                times [Array]: time array in observer frame
                nus [Array]: frequency array in observer frame
                log10_flux [Array]: Array of log10-fluxes in mJy.
        """
        x_array = jnp.array([x[name] for name in self.parameter_names])
        x_array = x_array.reshape(1,-1)
        x_tilde = self.X_scaler.transform(x_array)
        x_tilde = x_tilde.reshape(-1)
        x_tilde = jnp.concatenate((self.latent_vector, x_tilde))
        y = self.models.apply_fn({'params': self.models.params}, x_tilde)

        log10_flux = self.y_scaler.inverse_transform(y)
        log10_flux = log10_flux.reshape(len(self.nus), len(self.times))

        times, nus, flux = apply_redshift(10**log10_flux, self.times, self.nus, x.get("redshift", 0.0))
        log10_flux = jnp.log10(flux) - 10 - 2*jnp.log10(x.get("luminosity_distance", 1e-5))

        return times, nus, log10_flux



#--------------------------------------------------------
# DEPRECATED 
#--------------------------------------------------------

class LightcurveSurrogate(Surrogate):
    """Class of surrogate models that predicts the magnitudes per filter."""
    
    directory: str
    metadata: dict
    X_scaler: object
    y_scaler: dict[str, object]
    models: dict[str, TrainState]
    
    def __init__(
            self,
            name: str,
            filters: list[str],
            directory: str = None
        ) -> None:
        super().__init__(name, filters, directory)
        
        # Load the filters and networks
        self.load_filters(filters)
        self.load_networks()
        logger.info(f"Loaded surrogate {self.name} from {self.directory}. \n \n")
        logger.warning(
            "\n !!!! LightcurveSurrogate is no longer supported. "
            "Use our newer FluxSurrogates instead. "
            "Proceed at your own risk!!! \n"
        )
           
    def load_filters(self, filters_args: list[str] = None) -> None:
        # get all possible filters
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
        self.Filters = [fiesta_filters.Filter(filt) for filt in self.filters]
        logger.info(f"Surrogate {self.name} is loading with the following filters: {self.filters}.")
        
    def load_networks(self) -> None:
        pkl_files = [file for file in os.listdir(self.directory) if file.endswith(".pkl") or file.endswith(".pickle")]
        self.models = {}

        for filename in pkl_files:

            filter_of_filename = filename.split(".")[0].split("_")[1]

            if filter_of_filename in self.filters:
                state, _ = fiesta_nn.MLP.load_model(os.path.join(self.directory, filename))
                self.models[filter_of_filename] = state
    
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
