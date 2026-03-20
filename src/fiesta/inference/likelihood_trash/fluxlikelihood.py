import copy
from typing import Callable

import numpy as np
import jax
from jaxtyping import Float, Array
import jax.numpy as jnp

from fiesta.inference import LightcurveModel
from fiesta.inference.analytical_models import AnalyticalModel
from fiesta.logging import logger
from fiesta.inference.likelihood.base import LikelihoodBase

class FluxLikelihood(LikelihoodBase):

    """
    Likelihood object to compute likelihoods for the model parameters and a set of flux data points.
    Note that the ``data`` in the input argument still needs to be magnitudes. They will be converted internally to fluxes.

    Parameters
    ----------
    model: LightcurveModel | AnalyticalModel
        Light curve model that generates the estimated light curve from the parameters passed to ``evaluate``.
    data: dict[str, Float[Array, "ntimes 3"]]
        Dictionary with photometric filters as keys and arrays as values. 
        The first column of the array are the detection times in MJD.
        The second column the magnitude data points.
        The third column are the Gaussian measurement errors. 
        If an error is ``np.inf``, the data point will be treated as an upper limit on the light curve.
    trigger_time: Float
        Trigger time or start point of the light curve in MJD.
    data_tmin: Float
        Time point (in observer frame, relative to ``trigger_time``) before any data point from ``data`` will be cropped. Defaults to 0.0.
    data_tmax:
        Time point (in observer frame, relative to ``trigger_time``) after which any data point from ``data`` will be cropped. Defaults to 999.0
    filters: list[str]
        Filters that should be used for the likelihood evaluation. If None, will take filters from ``data``. Defaults to None.
    error_budget: Float
        Fixed error budget for the systematic uncertainty. Defaults to 1 mJy.
    conversion_function: Callable
        Function that will be called on the params before ``model`` predicts the light curve. Defaults to the idenity.
    fixed_params: dict[str, Float]
        Fixed parameters. These are added to the params before ``model`` predicts the light curve. Defaults to ``{}``.
    detection_limit: Float
        Detection limit of the telescope. If set, a truncated gaussian likelihood will be used. Defaults to None.
    zero_point_mag:
        Zero-point for mag-to-flux conversion, specifically to mJy (defaults to 16.4 for AB mag).


    Attributes
    ----------
    times_det: dict[str, Array]
        The time points of the detected fluxes per filter relative to the trigger time.
    times_nondet: dict[str, Array]
        The time points of the non-detected fluxes (upper limits) per filter relative to the trigger time.
    datapoints_det: dict[str, Array]
        The detected fluxs per filter.
    datapoints_nondet: dict[str, Array]
        The non-detection fluxes (upper limits) per filter.
    datapoints_err: dict[str, Array]
        The gaussian measurement error of the detected fluxes per filter.
    """

    def __init__(self,
                 model: LightcurveModel | AnalyticalModel,
                 data: dict[str, Float[Array, "ntimes 3"]],
                 trigger_time: Float,
                 data_tmin: Float = 0.0,
                 data_tmax: Float = 999.0,
                 filters: list[str] | None =  None,
                 error_budget: Float = 1,
                 conversion_function: Callable = lambda x: x,
                 fixed_params: dict[str, Float] = {},
                 detection_limit: Float = None,
                 zero_point_mag: Float = 16.4):

        super().__init__(model,
                         data,
                         trigger_time,
                         data_tmin,
                         data_tmax,
                         filters,
                         error_budget,
                         conversion_function,
                         fixed_params,
                         detection_limit)

        self.zero_point_mag = zero_point_mag

        self.datapoints_det = jax.tree.map(self.mag_to_flux, self.datapoints_det)
        self.datapoints_nondet = jax.tree.map(self.mag_to_flux, self.datapoints_nondet)
        # TODO: the next line is Gaussian error propagation, can we do something better?
        self.datapoints_err = 0.4 * jnp.log(10) * jax.tree.map(lambda x,y : x*y, self.datapoints_det, self.datapoints_err)

        logger.info("Loading and preprocessing observations in likelihood . . . DONE")

    def mag_to_flux(self, mag_arr: Array):
        """Converts mag_arr to fluxes in mJy."""
        return jnp.power(10, -0.4*(mag_arr-self.zero_point_mag))

    def evaluate(self, theta: dict[str, Array]) -> Float:
        """
        Evaluate the log-likelihood of the data given the model and the parameters theta, at a single point.

        Args:
            theta (dict[str, Array]): A dictionary containing the parameters used to generate the model light curve that is then used to compute the loglikelihood.

        Returns:
            Float: The log-likelihood value at this parameter point.
        """

        theta = {**theta, **self.fixed_params}
        theta = self.conversion(theta)
        times, mag_app = self.model.predict(theta)
        
        # Interpolate the mags to the times of interest
        mag_est_det = jax.tree.map(
            lambda t, m: jnp.interp(t, times, m, left = "extrapolate", right = "extrapolate"),
            self.times_det, mag_app
        )
        flux_est_det = jax.tree.map(self.mag_to_flux, mag_est_det)
        
        mag_est_nondet = jax.tree.map(
            lambda t, m: jnp.interp(t, times, m, left = "extrapolate", right = "extrapolate"),
            self.times_nondet, mag_app
        )
        flux_est_nondet = jax.tree.map(self.mag_to_flux, mag_est_nondet)
        
        # Get the systematic uncertainty + data uncertainty
        sigma = self.get_sigma(theta)
        nondet_sigma = self.get_nondet_sigma(theta)
        
        # Get likelihood from detections
        logl_det = jax.tree.map(
            self.get_gaussprob_det,
            flux_est_det,
            self.datapoints_det,
            sigma,
            self.detection_limit
        )
        logl_det_flat, _ = jax.flatten_util.ravel_pytree(logl_det)
        logl_det_total = jnp.sum(logl_det_flat)
        
        # Get likelihood from non-detections:
        logl_nondet = jax.tree_util.tree_map(
            self.get_gaussprob_nondet,
            flux_est_nondet,
            self.datapoints_nondet,
            nondet_sigma
        )
        logl_nondet_flat, _ = jax.flatten_util.ravel_pytree(logl_nondet)
        logl_nondet_total = jnp.sum(logl_nondet_flat)
        
        return logl_det_total + logl_nondet_total