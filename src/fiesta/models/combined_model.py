"""Class to combine several separate models."""
from functools import partial

from jaxtyping import Float, Array
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from fiesta.logging import logger
from fiesta.filters import Filter
from .base import FiestaModel

class CombinedModel(FiestaModel):
    def __init__(
            self,
            models: list[FiestaModel],
            sample_times: Array
        ):
        """
        API to combine several models in to one object. 
        Predicts the joined light curve from the surrogates listed in ``models``.

        Args:
            models (list[SurrogateModel]): A list of the surrogates that should be combined.
            sample_times (Array): (jax)-numpy array for the observer frame time at which the joint emission should be computed.
                                  Can reach beyond the time range of the individual surrogates, in which case the light curve will be extrapolated to the first value (left) or jnp.inf (right).
        """

        self.models = models
        self.times = jnp.array(sample_times)
        self.parameter_names = list(dict.fromkeys(
             p for model in self.models for p in model.parameter_names
         ))
        self._load_filters()

        logger.info(f"Initialized {self} with observer frame time range [ {self.times[0].item():.2f} {self.times[-1].item():.2f}] days.")
        for model in self.models:
            logger.info(f"\t {model} contributes in {model.filters}.")
    
    def _load_filters(self,):
        filters = []
        for model in self.models:
            filters.extend(model.filters)
        
        self.filters = list(set(filters))
        self.Filters = [Filter(filt) for filt in self.filters]

        self.add_filters(self.filters)
    
    @partial(jax.jit, static_argnums=(0,))
    def predict(self, x: dict[str, Array]) -> tuple[Array, dict[str, Array]]:

        """
        Predict the joint light curve by combining several separate submodels.

        Args:
            x (dict[str, Array]): Input array, unnormalized and untransformed.
                                  All model parameters from all models need to be specified here.
        
        Returns:
            tuple:
                times (Array): time array in observer frame
                mag (dict[str, Array]): The predicted magnitudes per filter
        """

        def predict_per_model(model):
            times, mags = model.predict(x)
            mag_interp = jax.tree.map(lambda mag: jnp.interp(self.times, times, mag, right=jnp.inf), mags)
            return mag_interp
        mag_dicts = jax.tree.map(predict_per_model, self.models)
        
        def add_magnitudes(filt):
            filt_mags = jnp.array([_dic.get(filt, jnp.ones_like(self.times)*jnp.inf) for _dic in mag_dicts])
            total_mag = -2.5 / jnp.log(10) * logsumexp(-jnp.log(10) / 2.5 * filt_mags, axis=0)
            return total_mag
        added_mags = jax.tree.map(add_magnitudes, self.filters)

        return self.times, dict(zip(self.filters, added_mags))
    
    def add_filters(self, filters: list[str] | str | Filter):
        for model in self.models:
            model.add_filters(filters)

        # only add filters here that have actually been added to at least one model
        super().add_filters(
            [filt for model in self.models for filt in model.filters]
        )
    
    def __repr__(self):
        return f"CombinedModel({[model.name for model in self.models]})"