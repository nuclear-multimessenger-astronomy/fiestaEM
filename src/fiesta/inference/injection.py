"""Functions for creating and handling injections"""
# TODO: for now, we will only support creating injections from a given model

import argparse
import copy
import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

from fiesta.inference.lightcurve_model import LightcurveModel
from fiesta.utils import mag_app_from_mag_abs

# TODO: get the parser going
def get_parser(**kwargs):
    add_help = kwargs.get("add_help", True)

    parser = argparse.ArgumentParser(
        description="Inference on kilonova and GRB parameters.",
        add_help=add_help,
    )
    

class InjectionRecovery:
    
    def __init__(self, 
                 model: LightcurveModel,
                 filters: list[str],
                 injection_dict: dict[str, Float],
                 tmin: Float = 0.1,
                 tmax: Float = 14.0,
                 N_datapoints: int = 10,
                 error_budget: Float = 1.0,):
        
        self.model = model
        # Ensure given filters are also in the trained model
        for filt in filters:
            if filt not in model.filters:
                print(f"Filter {filt} not in model filters. Removing from list")
                filters.remove(filt)
        self.filters = filters
        self.injection_dict = injection_dict
        self.tmin = tmin
        self.tmax = tmax
        self.N_datapoints = N_datapoints
        self.error_budget = error_budget
        
    def create_injection(self):
        """Create a synthetic injection from the given model and parameters."""
        
        self.data = {}
        all_mag_abs = self.model.predict(self.injection_dict)
        
        for filt in self.filters:
            times = self.create_timegrid()
            all_mag_app = mag_app_from_mag_abs(all_mag_abs[filt], self.injection_dict["luminosity_distance"])
            mag_app = np.interp(times, self.model.times, all_mag_app)
            mag_err = self.error_budget * np.ones_like(times)
            
            array = np.array([times, mag_app, mag_err]).T
            self.data[filt] = array
    
    def create_timegrid(self):
        """Create a time grid for the injection."""
        
        # TODO: create more interesting grids than uniform and same accross all filters?
        return np.linspace(self.tmin, self.tmax, self.N_datapoints)