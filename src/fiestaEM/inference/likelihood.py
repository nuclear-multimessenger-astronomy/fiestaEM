"""Functions for computing likelihoods of data given a model."""

import copy
import numpy as np
import jax
from jaxtyping import Float, Array
import jax.numpy as jnp

from fiestaEM.inference.lightcurve_model import LightcurveModel
from fiestaEM.utils import truncated_gaussian
from fiestaEM.conversions import mag_app_from_mag_abs

class EMLikelihood:
    
    model: LightcurveModel
    filters: list[str]
    trigger_time: Float
    tmin: Float
    tmax: Float
    
    detection_limit: dict[str, Array]
    error_budget: dict[str, Array]
    
    times_det: dict[str, Array]
    mag_det: dict[str, Array]
    mag_err: dict[str, Array]
    sigma: dict[str, Array]
    
    times_nondet: dict[str, Array]
    mag_nondet: dict[str, Array]
    
    def __init__(self, 
                 model: LightcurveModel, 
                 data: dict[str, Float[Array, "ntimes 3"]],
                 filters: list[str] = None,
                 trigger_time: Float = 0.0,
                 tmin: Float = 0.0,
                 tmax: Float = 999.0,
                 error_budget: Float = 1.0,
                 fixed_params: dict[str, Float] = {},
                 detection_limit: Float = None):
        
        # Save as attributes
        self.model = model
        if filters is None:
            filters = model.filters
        self.filters = filters
        self.trigger_time = trigger_time
        self.tmin = tmin
        self.tmax = tmax
        
        # Process error budget
        if isinstance(error_budget, (int, float)) and not isinstance(error_budget, dict):
            print("Converting error budget to dictionary.")
            error_budget = dict(zip(filters, [error_budget] * len(filters)))
        self.error_budget = error_budget
        
        # Process detection limit
        if isinstance(detection_limit, (int, float)) and not isinstance(detection_limit, dict):
            print("Converting detection limit to dictionary.")
            detection_limit = dict(zip(filters, [detection_limit] * len(filters)))
        
        if detection_limit is None:
            print("NOTE: No detection limit is given. Putting it to infinity.")
            detection_limit = dict(zip(filters, [jnp.inf] * len(filters)))
            
        self.detection_limit = detection_limit
            
        # TODO: for times, need to do some cross-checking against the times of the model and raise warnings
            
        # Process the given data
        self.times_det = {}
        self.mag_det = {}
        self.mag_err = {}
        
        self.times_nondet = {}
        self.mag_nondet = {}
        
        print("Loading and preprocessing observations in likelihood . . .")
        
        processed_data = copy.deepcopy(data)
        processed_data = {k.replace(":", "_"): v for k, v in processed_data.items()}
        
        for filt in self.filters:
            if filt not in processed_data:
                print(f"NOTE: Filter {filt} not found in the data. Removing for inference.")
                self.filters.remove(filt)
                continue
            
            # Preprocess times before data selection
            times, mag, mag_err = processed_data[filt].T
            times -= self.trigger_time
            
            idx = np.where((times > self.tmin) * (times < self.tmax))[0]
            times, mag, mag_err = times[idx], mag[idx], mag_err[idx]
            
            # Get detections
            idx_no_inf = np.where(mag_err != np.inf)[0]
            self.times_det[filt] = times[idx_no_inf]
            self.mag_det[filt] = mag[idx_no_inf]
            self.mag_err[filt] = mag_err[idx_no_inf]
            
            # Get non-detections
            idx_is_inf = np.where(mag_err == np.inf)[0]
            self.times_nondet[filt] = times[idx_is_inf]
            self.mag_nondet[filt] = mag[idx_is_inf]
        
        # Create auxiliary data structures used in calculations
        self.sigma = {}
        for filt in self.filters:
            self.sigma[filt] = jnp.sqrt(self.mag_err[filt] ** 2 + self.error_budget[filt] ** 2)
            
        self.fixed_params = fixed_params
        
        # Sanity check:
        detection_present = any([len(self.times_det[filt]) > 0 for filt in self.filters])
        assert detection_present, "No detections found in the data. Please check your data."
        print("Loading and preprocessing observations in likelihood . . . DONE")
        
    def __call__(self, theta):
        return self.evaluate(theta)
        
    def evaluate(self, 
                 theta: dict[str, Array],
                 data: dict = None) -> Float:
        """
        Evaluate the log-likelihood of the data given the model and the parameters theta, at a single point.

        Args:
            theta (dict[str, Array]): _description_
            data (dict, optional): Unused, but kept to comply with flowMC likelihood function signature. Defaults to None.

        Returns:
            Float: The log-likelihood value at this point.
        """
        
        theta = {**theta, **self.fixed_params}
        mag_abs: dict[str, Array] = self.model.predict(theta)
        mag_app = jax.tree.map(lambda x: mag_app_from_mag_abs(x, theta["luminosity_distance"]),
                               mag_abs)
        
        # Interpolate the mags to the times of interest
        mag_est_det = jax.tree.map(lambda t, m: jnp.interp(t, self.model.times, m),
                                          self.times_det, mag_app)
        
        mag_est_nondet = jax.tree.map(lambda t, m: jnp.interp(t, self.model.times, m),
                                          self.times_nondet, mag_app)
        
        # Get chisq
        chisq = jax.tree.map(self.get_chisq_filt, 
                             mag_est_det, self.mag_det, self.sigma, self.detection_limit)
        chisq_flatten, _ = jax.flatten_util.ravel_pytree(chisq)
        chisq_total = jnp.sum(chisq_flatten).astype(jnp.float64)
        
        # Get gaussprob:
        gaussprob = jax.tree.map(self.get_gaussprob_filt, 
                                 mag_est_nondet, self.mag_nondet, self.error_budget)
        gaussprob_flatten, _ = jax.flatten_util.ravel_pytree(gaussprob)
        gaussprob_total = jnp.sum(gaussprob_flatten).astype(jnp.float64)
        
        return chisq_total + gaussprob_total
    
    ### LIKELIHOOD FUNCTIONS ###
    
    def get_chisq_filt(self,
                       mag_est: Array,
                       mag_det: Array,
                       sigma: Array,
                       lim: Float) -> Float:
        """
        Return the log likelihood of the chisquare part of the likelihood function for a single filter.
        Branch-off of jax.lax.cond is based on provided detection limit (lim). If the limit is infinite, the likelihood is calculated without truncation and without resorting to scipy for faster evaluation. If the limit is finite, the likelihood is calculated with truncation and with scipy. 
        TODO: can we circumvent using scipy and implement this ourselves to speed up?

        Args:
            mag_est (Array): The estimated apparent magnitudes at the detection times
            mag_det (Array): The detected apparent magnitudes
            sigma (Array): The uncertainties on the detected apparent magnitudes, including the error budget.
            lim (Float): The detection limit for this filter

        Returns:
            Float: The chi-square value for this filter
        """
        
        return jax.lax.cond(lim == jnp.inf,
                           lambda x: self.compute_chisq(*x),
                           lambda x: self.compute_chisq_trunc(*x),
                           (mag_est, mag_det, sigma, lim))
    
    @staticmethod
    def compute_chisq(mag_est: Array,
                      mag_det: Array,
                      sigma: Array,
                      lim: Float) -> Float:
        """
        Return the log likelihood of the chisquare part of the likelihood function, without truncation (no detection limit is given), i.e. a Gaussian pdf. See get_chisq_filt for more details.
        """
        val = - 0.5 * jnp.sum(
            (mag_det - mag_est) ** 2 / sigma ** 2
        )
        return val
    
    @staticmethod
    def compute_chisq_trunc(mag_est: Array,
                            mag_det: Array,
                            sigma: Array,
                            lim: Float) -> Float:
        """
        Return the log likelihood of the chisquare part of the likelihood function, with truncation of the Gaussian (detection limit is given). See get_chisq_filt for more details.
        """
        return jnp.sum(truncated_gaussian(mag_det, sigma, mag_est, lim))
        
    def get_gaussprob_filt(self,
                           mag_est: Array,
                           mag_nondet: Array,
                           error_budget: Float) -> Float:
        
        return jax.lax.cond(len(mag_est) == 0,
                           lambda x: 0.0,
                           lambda x: self.compute_gaussprob(*x),
                           (mag_est, mag_nondet, error_budget))
        
    @staticmethod
    def compute_gaussprob(mag_est: Array,
                          mag_nondet: Array,
                          error_budget: Float) -> Float:
        gausslogsf = jax.scipy.stats.norm.logsf(
                    mag_nondet, mag_est, error_budget
                    )
        return jnp.sum(gausslogsf)