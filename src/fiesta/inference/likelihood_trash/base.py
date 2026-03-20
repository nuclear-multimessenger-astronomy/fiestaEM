"""Functions for computing likelihoods of data given a model."""

import copy
from typing import Callable

import numpy as np
import jax
from jaxtyping import Float, Array
import jax.numpy as jnp

from fiesta.inference import LightcurveModel
from fiesta.inference.analytical_models import AnalyticalModel
from fiesta.utils import truncated_gaussian
from fiesta.logging import logger

class LikelihoodBase:

    """
    Base class for likelihoods.
    """
    
    model: LightcurveModel | AnalyticalModel
    filters: list[str]
    trigger_time: Float
    data_tmin: Float
    data_tmax: Float
    
    detection_limit: dict[str, Array]
    error_budget: dict[str, Array]

    times_det: dict[str, Array]
    times_nondet: dict[str, Array]
    
    datapoints_det: dict[str, Array]
    datapoints_nondet: dict[str, Array]
    datapoints_err: dict[str, Array]

    
    def __init__(self, 
                 model: LightcurveModel | AnalyticalModel,
                 data: dict[str, Float[Array, "ntimes 3"]],
                 trigger_time: Float,
                 data_tmin: Float = 0.0,
                 data_tmax: Float = 999.0,
                 filters: list[str] | None =  None,
                 error_budget: Float = 0.3,
                 conversion_function: Callable = lambda x: x,
                 fixed_params: dict[str, Float] = {},
                 detection_limit: Float = None):

        # Process the given data
        logger.info("Loading and preprocessing observations in likelihood . . .")

        # Save as attributes
        self.model = copy.deepcopy(model)
        self.conversion = conversion_function
        self.fixed_params = fixed_params
        self.trigger_time = trigger_time
        
        data = copy.deepcopy(data)
        processed_data = self.setup_filters_and_data(filters, data)
        processed_data = self.cut_data_to_time_range(data, data_tmin, data_tmax)

        self.times_det = {}
        self.datapoints_det = {}
        self.datapoints_err = {}
        
        self.times_nondet = {}
        self.datapoints_nondet = {}

        for filt in processed_data:

            times, datapoints, datapoints_err = processed_data[filt].T

            # Get detections
            idx_no_inf = ~np.isinf(datapoints_err)
            self.times_det[filt] = jnp.array(times[idx_no_inf])
            self.datapoints_det[filt] = jnp.array(datapoints[idx_no_inf])
            self.datapoints_err[filt] = jnp.array(datapoints_err[idx_no_inf])
            
            # Get non-detections
            idx_is_inf = np.isposinf(data)
            self.times_nondet[filt] = jnp.array(times[idx_is_inf])
            self.datapoints_nondet[filt] = jnp.array(datapoints[idx_is_inf])
    
        # Sanity check:
        detection_present = np.any([self.times_det[filt].size > 0 for filt in self.filters])
        assert detection_present, "No detections found in the data. Please check your data, data_tmin, and data_tmax."


        # Process error budget, only for default behavior
        # Fiesta class will overwrite the systematic uncertainty setup later
        self._setup_sys_uncertainty_fixed(error_budget=error_budget)

        # Process detection limit
        if isinstance(detection_limit, (int, float)) and not isinstance(detection_limit, dict):
            detection_limit = dict(zip(filters, [detection_limit] * len(self.filters)))
        
        if detection_limit is None:
            logger.info("No detection limit is given. Putting it to infinity.")
            detection_limit = dict(zip(self.filters, [jnp.inf] * len(self.filters)))
        
        self.detection_limit = detection_limit
    
    def setup_filters_and_data(self, 
                               filters: list[str], 
                               data: dict[str, Float[Array, "ntimes 3"]]) -> dict[str, Float[Array, "ntimes 3"]]:
        if filters is None:
            self.filters = list(data.keys())
        else:
            self.filters = []
            for filt in filters:
                if filt in data:
                    self.filters.append(filt)
                else:
                    logger.warning(f"Filter {filt} from likelihood argument not in data. Ignoring for inference.")
                    continue
        
        missing = [filt for filt in self.model.filters if filt not in data]
        for filt in missing:
            logger.warning(f'Filter {filt} from likelihood.model not in data. Removing from model for inference.')
            self.model.filters.remove(filt)
        
        processed_data = copy.deepcopy(data)
        for filt in data.keys():
            if filt not in self.filters:
                logger.warning(f"Filter {filt} from data not found in likelihood.filters. Removing from data for inference.")
                del processed_data[filt]
            elif filt not in self.model.filters:
                logger.warning(f"Filter {filt} from data not found in likelihood.model.filters. Removing from data for inference.")
                del processed_data[filt]

        return processed_data
    
    def cut_data_to_time_range(self, data: dict[str, Float[Array, "ntimes 3"]], 
                               data_tmin: Float,
                               data_tmax: Float) -> dict[str, Float[Array, "ntimes 3"]]:
        
        self.data_tmin = data_tmin
        self.data_tmax = data_tmax

        for filt in data: 
            # Preprocess times before data selection
            times, y, y_err = data[filt].T
            times -= self.trigger_time
            
            idx = self.data_tmin < times < self.data_tmax
            times, y, y_err = times[idx], y[idx], y[idx]

            data[filt] = jnp.stack([times, y, y_err]).T
        
        return data

    def _setup_sys_uncertainty_fixed(self, error_budget: dict | float | int):
        
        # fixed systematic uncertainty
        if isinstance(error_budget, (int, float)) and not isinstance(error_budget, dict):
            error_budget = dict(zip(self.filters, [error_budget] * len(self.filters)))
        self.error_budget = error_budget
        # Create auxiliary data structures used in calculations
        self.sigma = {}
        for filt in self.filters:
            self.sigma[filt] = jnp.sqrt(self.datapoints_err[filt] ** 2 + self.error_budget[filt] ** 2)

        self.get_sigma = lambda x: self.sigma
        self.get_nondet_sigma = lambda x: self.error_budget

    def _setup_sys_uncertainty_free(self,):

        # freely sampled sys. uncertainty, but same for all filters and times
        def _sigma(theta):
            sys_err = theta["em_syserr"]
            sigma = jax.tree.map(lambda mag_err: jnp.sqrt(mag_err**2 + sys_err**2), self.datapoints_err)
            return sigma
        
        def _nondet_sigma(theta):
            sigma = jax.tree.map(lambda mag_nondet: theta["em_syserr"], self.datapoints_nondet)
            return sigma
        
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
                                 self.datapoints_err,
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
        
    def evaluate(self, theta: dict[str, Array]) -> Float:
        """
        Evaluate the log-likelihood of the data given the parameters in theta and the underlying model.
        """
        raise NotImplementedError(f"Needs to be implemented by subclasses.")

    def vectorized_evaluate(self, theta: dict[str, Array]):

        theta_arr = jnp.array([theta[name] for name in theta.keys()]).T

        def evaluate_single(theta_single):
            param_dict = dict(zip(theta.keys(), theta_single))
            return self(param_dict)

        try:
            return jax.lax.map(evaluate_single, theta_arr, batch_size=500)
        except TypeError:
            # JAX < 0.5 doesn't support batch_size in lax.map
            return jax.vmap(evaluate_single)(theta_arr)
    
    ### LIKELIHOOD COMPUTATION METHODS ###

    # For detection data points
    #===========================

    def get_gaussprob_det(self,
                          y_est: Array,
                          y_data: Array,
                          sigma: Array,
                          lim: Float) -> Float:
        """
        Return the log likelihood of the gaussian likelihood function for a single filter.
        Branch-off of jax.lax.cond is based on provided detection limit (lim). 
        If the limit is infinite, the likelihood is calculated without truncation and without resorting to scipy for faster evaluation. 
        If the limit is finite, the likelihood is calculated with truncation and with scipy. 

        Args:
            y_est (Array): The estimated data from the model at detection times.
            y_data (Array): The detected data.
            sigma (Array): The uncertainties on the detected apparent magnitudes, including the error budget.
            lim (Float): The detection limit for this filter.

        Returns:
            Float: The gaussian log-likelihood for this filter.
        """
        return jax.lax.cond(lim == jnp.inf,
                           lambda x: self.compute_gaussian_likelihood(*x),
                           lambda x: self.compute_trunc_gaussian_likelihood(*x),
                           (y_est, y_data, sigma, lim))
    
    @staticmethod
    def compute_gaussian_likelihood(y_est: Array,
                                    y_data: Array,
                                    sigma: Array,
                                    lim: Float) -> Float:
        """
        Return the log likelihood of the chisquare part of the likelihood function, without truncation (no detection limit is given), i.e. a Gaussian pdf.
        """
        val = - 0.5 * jnp.sum( (y_data - y_est) ** 2 / sigma ** 2)
        val -= 1/2*jnp.sum(jnp.log(2*jnp.pi*sigma**2))
        return val

    @staticmethod
    def compute_trunc_gaussian_likelihood(y_est: Array,
                                          y_data: Array,
                                          sigma: Array,
                                          lim: Float) -> Float:
        """
        Return the log likelihood of the chisquare part of the likelihood function, with truncation of the Gaussian (detection limit is given).
        """
        return jnp.sum(truncated_gaussian(y_data, sigma, y_est, lim))


    # For non-detection data points
    #===========================

    
    def get_gaussprob_nondet(self,
                             y_est: Array,
                             y_data: Array,
                             error_budget: Float) -> Float:
        """
        Return the log likelihood of the gaussian likelihood function for a single filter.
        Branch-off of jax.lax.cond is based on provided detection limit (lim). 
        If the limit is infinite, the likelihood is calculated without truncation and without resorting to scipy for faster evaluation. 
        If the limit is finite, the likelihood is calculated with truncation and with scipy. 

        Args:
            y_est (Array): The estimated data from the model at detection times.
            y_data (Array): The nondetection data points.
            sigma (Array): The uncertainties on the detected apparent magnitudes, including the error budget.
            lim (Float): The detection limit for this filter.

        Returns:
            Float: The gaussian log-likelihood for this filter.
        """
        
        return jax.lax.cond(len(y_data) == 0,
                            lambda x: 0.0,
                            lambda x: self.compute_gaussian_survival(*x),
                            (y_est, y_data, error_budget))

    @staticmethod
    def compute_gaussian_survival(y_est: Array,
                                  y_data: Array,
                                  error_budget: Array) -> Float:
        
        gausslogsf = jax.scipy.stats.norm.logsf(y_data, y_est, error_budget)
        return jnp.sum(gausslogsf)