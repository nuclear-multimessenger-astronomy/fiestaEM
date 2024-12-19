"""Functions for creating and handling injections"""
# TODO: for now, we will only support creating injections from a given model

import argparse
import copy
import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

from fiestaEM.inference.lightcurve_model import LightcurveModel
from fiestaEM.conversions import mag_app_from_mag_abs
from fiestaEM.utils import Filter
from fiestaEM.constants import days_to_seconds, c
from fiestaEM import conversions

import afterglowpy as grb

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
        self.tmin = tmin
        self.tmax = tmax
        self.N_datapoints = N_datapoints
        self.error_budget = error_budget
        self.randomize_nondetections = randomize_nondetections
        self.randomize_nondetections_fraction = randomize_nondetections_fraction
        
    def create_injection(self):
        """Create a synthetic injection from the given model and parameters."""
        
        points = np.random.multinomial(self.N_datapoints, [1/len(self.filters)]*len(self.filters)) # random number of datapoints in each filter
        self.data = {}

        for npoints, filt in zip(points, self.filters):
            self.injection_dict["nu"] = filt.nu
            times = self.create_timegrid(npoints)
            mJys = self._call_afterglowpy(times*days_to_seconds, self.injection_dict)
            magnitudes = conversions.mJys_to_mag_np(mJys)
            mag_err = self.error_budget * np.ones_like(times)
            self.data[filt.name] = np.array([times, magnitudes, mag_err]).T

        
    def _call_afterglowpy(self,
                         times_afterglowpy: Array,
                         params_dict: dict[str, Float]) -> Float[Array, "n_times"]:
        """
        Call afterglowpy to generate a single flux density output, for a given set of parameters. Note that the parameters_dict should contain all the parameters that the model requires, as well as the nu value.
        The output will be a set of mJys.

        Args:
            Float[Array, "n_times"]: The flux density in mJys at the given times.
        """
        
        # Preprocess the params_dict into the format that afterglowpy expects, which is usually called Z
        Z = {}
        
        Z["jetType"]  = params_dict.get("jetType", self.jet_type)
        Z["specType"] = params_dict.get("specType", 0)
        Z["z"] = params_dict.get("z", 0.0)
        Z["xi_N"] = params_dict.get("xi_N", 1.0)
            
        Z["E0"]        = 10 ** params_dict["log10_E0"]
        Z["thetaCore"] = params_dict["thetaCore"]
        Z["n0"]        = 10 ** params_dict["log10_n0"]
        Z["p"]         = params_dict["p"]
        Z["epsilon_e"] = 10 ** params_dict["log10_epsilon_e"]
        Z["epsilon_B"] = 10 ** params_dict["log10_epsilon_B"]
        Z["d_L"]       = params_dict.get("luminosity_distance", 1e-5)*1e6*3.086e18
        if "inclination_EM" in list(params_dict.keys()):
            Z["thetaObs"]  = params_dict["inclination_EM"]
        else:
            Z["thetaObs"]  = params_dict["thetaObs"]
        if self.jet_type == 1 or self.jet_type == 4:
            Z["b"] = params_dict["b"]
        if "thetaWing" in list(params_dict.keys()):
            Z["thetaWing"] = params_dict["thetaWing"]
        
        # Afterglowpy returns flux in mJys
        mJys = grb.fluxDensity(times_afterglowpy, params_dict["nu"], **Z)
        return mJys
    
    
    def create_timegrid(self, npoints):
        """Create a time grid for the injection."""

        return np.linspace(self.tmin, self.tmax, npoints)