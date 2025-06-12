"""Functions for computing likelihoods of data given a model."""

import copy
from typing import Callable

import numpy as np
import jax
from jaxtyping import Float, Array
import jax.numpy as jnp

from fiesta.inference.lightcurve_model import LightcurveModel
from fiesta.utils import truncated_gaussian
from fiesta.logging import logger

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
                 error_budget: Float = 0.3,
                 conversion_function: Callable = lambda x: x,
                 fixed_params: dict[str, Float] = {},
                 detection_limit: Float = None):
        
        # Save as attributes
        self.model = model
        self.conversion = conversion_function
        if filters is None:
            filters = model.filters
        self.filters = list(filters)
        self.trigger_time = trigger_time
        self.tmin = tmin
        self.tmax = tmax            
        # TODO: for times, need to do some cross-checking against the times of the model and raise warnings
            
        # Process the given data
        logger.info("Loading and preprocessing observations in likelihood . . .")
        self.times_det = {}
        self.mag_det = {}
        self.mag_err = {}
        
        self.times_nondet = {}
        self.mag_nondet = {}
        
        processed_data = copy.deepcopy(data)
        filter_copy = self.filters.copy()

        for filt in filter_copy:
            if filt not in processed_data:
                logger.warning(f"Filter {filt} not found in the data. Removing for inference.")
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
        
        # Process detection limit
        if isinstance(detection_limit, (int, float)) and not isinstance(detection_limit, dict):
            logger.info("Converting detection limit to dictionary.")
            detection_limit = dict(zip(filters, [detection_limit] * len(filters)))
        
        if detection_limit is None:
            logger.info("No detection limit is given. Putting it to infinity.")
            detection_limit = dict(zip(filters, [jnp.inf] * len(filters)))
        
        self.detection_limit = detection_limit
        
        # Process error budget, only for default behavior
        # fiesta class will overwrite the systematic uncertainty setup later
        self._setup_sys_uncertainty_fixed(error_budget=error_budget)
                
        self.fixed_params = fixed_params
        
        # Sanity check:
        detection_present = any([len(self.times_det[filt]) > 0 for filt in self.filters])
        assert detection_present, "No detections found in the data. Please check your data."
        logger.info("Loading and preprocessing observations in likelihood . . . DONE")

    def _setup_sys_uncertainty_fixed(self, error_budget: dict | float | int):
        
        # fixed systematic uncertainty
        if isinstance(error_budget, (int, float)) and not isinstance(error_budget, dict):
            error_budget = dict(zip(self.filters, [error_budget] * len(self.filters)))
        self.error_budget = error_budget
        # Create auxiliary data structures used in calculations
        self.sigma = {}
        for filt in self.filters:
            self.sigma[filt] = jnp.sqrt(self.mag_err[filt] ** 2 + self.error_budget[filt] ** 2)

        self.get_sigma = lambda x: self.sigma
        self.get_nondet_sigma = lambda x: self.error_budget

    def _setup_sys_uncertainty_free(self,):

        # freely sampled sys. uncertainty, but same for all filters and times
        def _sigma(theta):
            sys_err = theta["sys_err"]
            sigma = jax.tree.map(lambda mag_err: jnp.sqrt(mag_err**2 + sys_err**2), self.mag_err)
            return sigma
        
        def _nondet_sigma(theta):
            return theta["sys_err"]
        
        self.get_sigma = _sigma
        self.get_nondet_sigma = _nondet_sigma

    def _setup_sys_uncertainty_from_file(self, 
                                         sys_params_per_filter: dict[str, list], 
                                         t_nodes_per_filter: dict[str, Array]):
        
        # systematic uncertainty setup from file
        self.sys_params_per_filter = sys_params_per_filter

        for key in t_nodes_per_filter:
            if t_nodes_per_filter[key] is None:
                t_nodes_per_filter[key] = jnp.linspace(self.tmin, self.tmax, len(self.sys_params_per_filter[key]))
        self.t_nodes_per_filter = t_nodes_per_filter

        def _get_sigma(theta):
            def add_sys_err(mag_err, time_det, params, t_nodes):
                sys_param_array = jnp.array([theta[p] for p in params])
                sigma_sys = jnp.interp(time_det, t_nodes, sys_param_array)
                return jnp.sqrt(sigma_sys**2 + mag_err **2)
            
            sigma = jax.tree.map(add_sys_err, 
                                 self.mag_err, 
                                 self.times_det, 
                                 self.sys_params_per_filter, 
                                 self.t_nodes_per_filter)
            return sigma
        
        def _nondet_sigma(theta):
            def fetch_sigma(time_nondet, params, t_nodes):
                sys_param_array = jnp.array([theta[p] for p in params])
                return jnp.interp(time_nondet, t_nodes, sys_param_array)

            sigma = jax.tree.map(fetch_sigma, 
                                 self.times_nondet, 
                                 self.sys_params_per_filter, 
                                 self.t_nodes_per_filter)
            return sigma
        
        self.get_sigma = _get_sigma
        self.get_nondet_sigma = _nondet_sigma


        
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
        theta = self.conversion(theta)
        times, mag_app = self.model.predict(theta)
        
        # Interpolate the mags to the times of interest
        mag_est_det = jax.tree_util.tree_map(lambda t, m: jnp.interp(t, times, m, left = "extrapolate", right = "extrapolate"), # TODO extrapolation is maybe problematic here
                                          self.times_det, mag_app)
        
        mag_est_nondet = jax.tree_util.tree_map(lambda t, m: jnp.interp(t, times, m, left = "extrapolate", right = "extrapolate"),
                                          self.times_nondet, mag_app)
        
        # Get the systematic uncertainty + data uncertainty
        sigma = self.get_sigma(theta)
        nondet_sigma = self.get_nondet_sigma(theta)
        
        # Get chisq
        chisq = jax.tree_util.tree_map(self.get_chisq_filt, 
                             mag_est_det, self.mag_det, sigma, self.detection_limit)
        chisq_flatten, _ = jax.flatten_util.ravel_pytree(chisq)
        chisq_total = jnp.sum(chisq_flatten)#.astype(jnp.float64)
        
        # Get gaussprob:
        gaussprob = jax.tree_util.tree_map(self.get_gaussprob_filt, 
                                 mag_est_nondet, self.mag_nondet, nondet_sigma)
        gaussprob_flatten, _ = jax.flatten_util.ravel_pytree(gaussprob)
        gaussprob_total = jnp.sum(gaussprob_flatten)#.astype(jnp.float64)
        
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
        val = - 0.5 * jnp.sum( (mag_det - mag_est) ** 2 / sigma ** 2) 
        val -= 1/2*jnp.sum(jnp.log(2*jnp.pi*sigma**2))
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
    
    def v_evaluate(self, theta: dict[str, Array]):

        def _evaluate_single(_theta):
            return self(_theta)
        return jax.vmap(_evaluate_single)(theta)
