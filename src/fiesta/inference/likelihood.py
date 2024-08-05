"""Functions for computing likelihoods of data given a model."""

import copy
import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

from fiesta.inference.lightcurve_model import LightcurveModel
from fiesta.utils import mag_app_from_mag_abs

class EMLikelihood:
    
    model: LightcurveModel
    filters: list[str]
    trigger_time: Float
    tmin: Float
    tmax: Float
    ignore_nondetections: bool
    detection_limit: Float
    error_budget: Float
    
    times_det: dict[str, Array]
    mag_det: dict[str, Array]
    mag_err_det: dict[str, Array]
    data_sigma: dict[str, Array]
    
    times_nondet: dict[str, Array]
    mag_nondet: dict[str, Array]
    
    def __init__(self, 
                 model: LightcurveModel, 
                 filters: list[str],
                 data: dict[str, Float[Array, "ntimes 3"]],
                 trigger_time: Float = 0.0,
                 tmin: Float = 0.0,
                 tmax: Float = 14.0,
                 error_budget: Float = 1.0,
                 fixed_params: dict[str, Float] = {},
                 ignore_nondetections: bool = True):
        # detection_limit: Float = 9999.0 # TODO: add detection limit as an argument
        
        # Save as attributes
        self.model = model
        self.filters = filters
        self.trigger_time = trigger_time
        self.tmin = tmin
        self.tmax = tmax
        self.ignore_nondetections = ignore_nondetections
        self.error_budget = error_budget # TODO: generalize this to be a dictionary of errors for each filter?
        # self.detection_limit = detection_limit # TODO: generalize this to be a dictionary of detection limits for each filter?
        
        # Process the given data
        self.times_det = {}
        self.mag_det = {}
        self.mag_err_det = {}
        
        self.times_nondet = {}
        self.mag_nondet = {}
        self.process_data(data)
        
        # If there are no non-detections, automatically ignore them below
        if len(self.times_nondet) == 0:
            print("NOTE: No non-detections found in the data. Ignoring non-detections.")
            self.ignore_nondetections = True
        
        # Create auxiliary data structures used in calculations
        self.data_sigma = {}
        for filt in self.filters:
            # TODO: generalize to dict for error_budget
            self.data_sigma[filt] = jnp.sqrt(self.mag_err_det[filt] ** 2 + self.error_budget ** 2)
            
        self.fixed_params = fixed_params
        
    def log_likelihood(self, 
                       theta: dict[str, Array]):
        
        # # Make sure the fixed parameters have the same PyTree structure
        # example_tree = theta[theta.keys()[0]]
        # treevals, treedef = jax.tree.flatten(example_tree)
        # for key, value in self.fixed_params.items():
        #     self.fixed_params[key] = jax.tree_util.tree_unflatten(treedef, jax.tree_util.tree_flatten(value)[0])
        
        # # Update theta dict with the fixed parameters as well
        # fixed_params_dict = {}
        # for key, value in self.fixed_params.items():
        #     # TODO: this is cumbersome...
        #     example_key = list(theta.keys())[0]
        #     fixed_params_dict[key] = value * jnp.ones_like(theta[example_key])
            
        theta = {**theta, **self.fixed_params}
        
        mag_abs = self.model.predict(theta) # dict[str, Array] comes out
        
        mag_app = jax.tree.map(lambda x: mag_app_from_mag_abs(x, theta["luminosity_distance"]),
                               mag_abs)
        
        # Interpolate the mags to the times of the detections
        mag_app_interp = jax.tree.map(lambda t, m: jnp.interp(t, self.model.times, m),
                                        self.times_det, mag_app)
        
        # Get chisq
        chisq = jax.tree.map(lambda mag_est, mag_det, data_sigma: self.get_chisq_filt(mag_est, mag_det, data_sigma), 
                             mag_app_interp, self.mag_det, self.data_sigma)
        chisq_flatten, _ = jax.flatten_util.ravel_pytree(chisq)
        chisq_total = jnp.sum(chisq_flatten)
        
        # TODO: implement the non-detections part of the likelihood
        gaussprob_total = jnp.array([0.0])
        
        return chisq_total + gaussprob_total
    
    def __call__(self, theta):
        return self.log_likelihood(theta)
    
    def process_data(self, data: dict[str, Array]):
        """
        Separate the data into the "detections" and "non-detections" categories and apply necessary masking and other desired preprocessing steps.

        Args:
            data (np.array): Input array of shape (n, 3) where n is the number of data points and the columns are times, magnitude, and magnitude error.

        Returns:
            None. Sets the desired attributes. See source code for details.
        """
        
        print("Loading and preprocessing observations . . .")
        
        processed_data = copy.deepcopy(data)
        processed_data = {k.replace(":", "_"): v for k, v in processed_data.items()}
        
        for filt in self.filters:
            if filt not in processed_data:
                print(f"NOTE: Filter {filt} not found in the data. Removing for inference.")
                self.filters.remove(filt)
                continue
            
            times, mag, mag_err = processed_data[filt][:, 0], processed_data[filt][:, 1], processed_data[filt][:, 2]
            times -= self.trigger_time
            idx = np.where((times > self.tmin) * (times < self.tmax))[0]
            times, mag, mag_err = times[idx], mag[idx], mag_err[idx]
            
            # TODO: Apply the detection limit here?
            idx_no_inf = np.where(mag_err != np.inf)[0]
            
            self.times_det[filt] = times[idx_no_inf]
            self.mag_det[filt] = mag[idx_no_inf]
            self.mag_err_det[filt] = mag_err[idx_no_inf]
            
            self.times_nondet[filt] = times[~idx_no_inf]
            self.mag_nondet[filt] = mag[~idx_no_inf]
            
    ### LIKELIHOOD FUNCTIONS ###
    
    def get_chisq_filt(self,
                       mag_est: Array,
                       mag_det: Array,
                       data_sigma: Array,
                       timeshift: Float = 0.0,
                    #    upper_lim: Float = 9999.0, # TODO: implement this!
                    #    lower_lim: Float = -9999.0
                    ) -> Float:
        """
        Return the log likelihood of the chisquare part of the likelihood function.

        Args:
            mag_est (Array): Estimated magnitudes for this filter
            timeshift (Float, optional): Timeshift to be applied. Defaults to 0.0. TODO: has to be implemented properly

        Returns:
            Float: The chi-square value for this filter
        """
        
        arg = - 0.5 * jnp.sum(
            (mag_det - mag_est) ** 2 / data_sigma ** 2
        )
        return arg
        
        # TODO: implement this!
        # minus_chisquare = jnp.sum(
        #     truncated_gaussian(
        #         data_mag,
        #         data_sigma,
        #         mag_est,
        #         upper_lim=upper_lim,
        #         lower_lim=lower_lim,
        #     )
        # )
        # return minus_chisquare