"""Functions for creating and handling injections"""
# TODO: for now, we will only support creating injections from a given model

import argparse
import copy
import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

from fiesta.inference.lightcurve_model import LightcurveModel
from fiesta.conversions import mag_app_from_mag_abs
from fiesta.utils import Filter
from fiesta.constants import days_to_seconds, c
from fiesta import conversions

from fiesta.train.AfterglowData import RunAfterglowpy

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
                 injection_dict: dict[str, Float],
                 filters: list[str] = None,
                 tmin: Float = 0.1,
                 tmax: Float = 14.0,
                 N_datapoints: int = 10,
                 error_budget: Float = 1.0,
                 randomize_nondetections: bool = False,
                 randomize_nondetections_fraction: Float = 0.2):
        
        self.model = model
        # Ensure given filters are also in the trained model
        if filters is None:
            filters = model.filters
        else:
            for filt in filters:
                if filt not in model.filters:
                    print(f"Filter {filt} not in model filters. Removing from list")
                    filters.remove(filt)
     
        print(f"Creating injection with filters: {filters}")
        self.filters = filters
        self.injection_dict = injection_dict
        self.tmin = tmin
        self.tmax = tmax
        self.N_datapoints = N_datapoints
        self.error_budget = error_budget
        self.randomize_nondetections = randomize_nondetections
        self.randomize_nondetections_fraction = randomize_nondetections_fraction
        
    def create_injection(self):
        """Create a synthetic injection from the given model and parameters."""
        
        self.data = {}
        all_mag_abs = self.model.predict(self.injection_dict)
        
        for filt in self.filters:
            times = self.create_timegrid()
            all_mag_app = mag_app_from_mag_abs(all_mag_abs[filt], self.injection_dict["luminosity_distance"])
            mag_app = np.interp(times, self.model.times, all_mag_app)
            mag_err = self.error_budget * np.ones_like(times)
            
            # Randomize to get some non-detections if so desired:
            if self.randomize_nondetections:
                n_nondetections = int(self.randomize_nondetections_fraction * len(times))
                nondet_indices = np.random.choice(len(times), size = n_nondetections, replace = False)
                
                mag_app[nondet_indices] -= 5.0 # randomly bump down the magnitude
                mag_err[nondet_indices] = np.inf
            
            array = np.array([times, mag_app, mag_err]).T
            self.data[filt] = array
    
    def create_timegrid(self):
        """Create a time grid for the injection."""
        
        # TODO: create more interesting grids than uniform and same accross all filters?
        return np.linspace(self.tmin, self.tmax, self.N_datapoints)



class InjectionRecoveryAfterglowpy:
    
    def __init__(self,
                 injection_dict: dict[str, Float],
                 trigger_time: Float,
                 filters: list[str],
                 jet_type = -1,
                 tmin: Float = 0.1,
                 tmax: Float = 1000.0,
                 N_datapoints: int = 10,
                 error_budget: Float = 1.0,
                 randomize_nondetections: bool = False,
                 randomize_nondetections_fraction: Float = 0.2):
        
        self.jet_type = jet_type
        # Ensure given filters are also in the trained model
        
        if filters is None:
            filters = model.filters

        self.filters = [Filter(filt) for filt in filters]
        print(f"Creating injection with filters: {filters}")
        self.injection_dict = injection_dict
        self.trigger_time = trigger_time
        self.tmin = tmin
        self.tmax = tmax
        self.N_datapoints = N_datapoints
        self.error_budget = error_budget
        self.randomize_nondetections = randomize_nondetections
        self.randomize_nondetections_fraction = randomize_nondetections_fraction
        
    def create_injection(self):
        """Create a synthetic injection from the given model and parameters."""
        
        nus = [filt.nu for filt in self.filters]
        times = np.logspace(np.log10(self.tmin), np.log10(self.tmax), 200)
        afgpy = RunAfterglowpy(self.jet_type, times, nus, [list(self.injection_dict.values())], self.injection_dict.keys())
        _, log_flux = afgpy(0)
        mJys  = np.exp(log_flux).reshape(len(nus), 200)

        self.data = {}
        points = np.random.multinomial(self.N_datapoints, [1/len(self.filters)]*len(self.filters)) # random number of datapoints in each filter
        for j, npoints, filt in zip(range(len(self.filters)), points, self.filters):
            times_data = self.create_timegrid(npoints)
            mJys_filter = np.interp(times_data, times, mJys[j])
            magnitudes = conversions.mJys_to_mag_np(mJys_filter)
            magnitudes = magnitudes + 5 * np.log10(self.injection_dict["luminosity_distance"]/(10*1e-6))

            mag_err = self.error_budget * np.ones_like(times_data)
            self.data[filt.name] = np.array([times_data + self.trigger_time, magnitudes, mag_err]).T 
    
    def create_timegrid(self, npoints):
        """Create a time grid for the injection."""

        return np.linspace(self.tmin, self.tmax, npoints)