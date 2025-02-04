"""Functions for creating and handling injections"""
# TODO: for now, we will only support creating injections from a given model

import argparse

import h5py
from jaxtyping import Float, Array
import numpy as np

from fiesta.inference.lightcurve_model import LightcurveModel
from fiesta.conversions import mag_app_from_mag_abs, apply_redshift
from fiesta.utils import Filter

from fiesta.train.AfterglowData import RunAfterglowpy, RunPyblastafterglow

# TODO: get the parser going
def get_parser(**kwargs):
    add_help = kwargs.get("add_help", True)

    parser = argparse.ArgumentParser(
        description="Inference on kilonova and GRB parameters.",
        add_help=add_help,
    )


class InjectionBase:

    def __init__(self,
                 filters: list[str],
                 trigger_time: float,
                 tmin: Float = 0.1,
                 tmax: Float = 1000.0,
                 N_datapoints: int = 10,
                 t_detect: dict[str, Array] = None,
                 error_budget: Float = 1.0,
                 nondetections: bool = False,
                 nondetections_fraction: Float = 0.2):
        
        self.Filters = [Filter(filt) for filt in filters]
        print(f"Creating injection with filters: {filters}")
        self.trigger_time = trigger_time

        if t_detect is not None:
           self.t_detect = t_detect 
        else:
            self.create_t_detect(tmin, tmax, N_datapoints)

        self.error_budget = error_budget
        self.nondetections = nondetections
        self.nondetections_fraction = nondetections_fraction
    
    def create_t_detect(self, tmin, tmax, N):
        """Create a time grid for the injection data."""

        self.t_detect = {}
        points_list = np.random.multinomial(N, [1/len(self.Filters)]*len(self.Filters)) # random number of time points in each filter

        for points, Filt in zip(points_list, self.Filters):
            t = np.exp(np.random.uniform(np.log(tmin), np.log(tmax), size = points))
            t = np.sort(t)
            t[::2] *= np.random.uniform(1, (tmax/tmin)**(1/points), size = len(t[::2])) # correlate the time points
            t[::3] *= np.random.uniform(1, (tmax/tmin)**(1/points), size = len(t[::3])) # correlate the time points
            t = np.minimum(t, tmax)
            self.t_detect[Filt.name] = np.sort(t)
    
    def create_injection(self,
                         injection_dict: dict[str, Float]):
        raise NotImplementedError
    
    def randomize_nondetections(self,):
        if not self.nondetections:
            return
        
        N = np.sum([len(self.t_detect[Filt.name]) for Filt in self.Filters])
        nondets_list = np.random.multinomial(int(N*self.nondetections_fraction), [1/len(self.Filters)]*len(self.Filters)) # random number of non detections in each filter

        for nondets, Filt in zip(nondets_list, self.Filters):
            inds = np.random.choice(np.arange(len(self.data[Filt.name])), size=nondets, replace=False)
            self.data[Filt.name][inds] += np.array([0, -5., np.inf])


        
    

class InjectionSurrogate(InjectionBase):
    
    def __init__(self, 
                 model: LightcurveModel,
                 *args,
                 **kwargs):
        
        self.model = model
        super().__init__(*args, **kwargs)
        
    def create_injection(self, injection_dict):
        """Create a synthetic injection from the given model and parameters."""

        injection_dict["luminosity_distance"] = injection_dict.get('luminosity_distance', 1e-5)
        injection_dict["redshift"] = injection_dict.get('redshift', 0)
        
        times, mags = self.model.predict(injection_dict)
        self.data = {}

        for Filt in self.Filters:
            t_detect = self.t_detect[Filt.name]

            mag_app = np.interp(t_detect, times, mags[Filt.name])

            mag_err = self.error_budget * np.sqrt(np.random.chisquare(df=1, size = len(t_detect)))
            mag_err = np.maximum(mag_err, 0.01)
            mag_err = np.minimum(mag_err, 1)
            
            array = np.array([t_detect, mag_app, mag_err]).T
            self.data[Filt.name] = array
        
        self.randomize_nondetections()

class InjectionAfterglowpy(InjectionBase):
    
    def __init__(self,
                 jet_type: int = -1,
                 *args,
                 **kwargs):
        
        self.jet_type = jet_type
        super().__init__(*args, **kwargs)
        
    def create_injection(self, injection_dict):
        """Create a synthetic injection from the given model and parameters."""

        nus = [nu for Filter in self.Filters for nu in Filter.nus]
        times = [t for Filter in self.Filters for t in self.t_detect[Filter.name]]

        nus = np.sort(nus)
        times = np.sort(times)

        afgpy = RunAfterglowpy(self.jet_type, times, nus, [list(injection_dict.values())], injection_dict.keys())
        _, log_flux = afgpy(0)
        mJys  = np.exp(log_flux).reshape(len(nus), len(times))

        self.data = {}

        for Filter in self.Filters:
            t_detect = self.t_detect[Filter.name]

            mag_abs = Filter.get_mag(mJys, nus) # even when 'luminosity_distance' is passed to RunAfterglowpy, it will return the abs mag (with redshift)
            mag_app = mag_app_from_mag_abs(mag_abs, injection_dict["luminosity_distance"])
            mag_app = np.interp(t_detect, times, mag_app)

            mag_err = self.error_budget * np.sqrt(np.random.chisquare(df=1, size = len(t_detect)))
            mag_err = np.maximum(mag_err, 0.01)
            mag_err = np.minimum(mag_err, 1)

            self.data[Filter.name] = np.array([t_detect + self.trigger_time, mag_app, mag_err]).T
        
        self.randomize_nondetections()

class InjectionPyblastafterglow(InjectionBase):
    
    def __init__(self,
                 jet_type: str = "tophat",
                 *args,
                 **kwargs):
        
        self.jet_type = jet_type
        super().__init__(*args, **kwargs)
        
    def create_injection(self, injection_dict):
        """Create a synthetic injection from the given model and parameters."""

        nus = [nu for Filter in self.Filters for nu in Filter.nus]
        times = [t for Filter in self.Filters for t in self.t_detect[Filter.name]]

        nus = np.sort(nus)
        times = np.sort(times)
        nus = np.logspace(np.log10(nus[0]), np.log10(nus[-1]), len(nus)) #pbag only takes log (or linear) spaced arrays
        nus = np.logspace(np.log10(times[0]), np.log10(times[-1]), len(times))

        pbag = RunPyblastafterglow(self.jet_type, times, nus, [list(injection_dict.values())], injection_dict.keys())
        _, log_flux = pbag(0)
        mJys  = np.exp(log_flux).reshape(len(nus), len(times))

        self.data = {}

        for Filter in self.Filters:
            t_detect = self.t_detect[Filter.name]

            mag_abs = Filter.get_mag(mJys, nus)
            mag_app = mag_app_from_mag_abs(mag_abs, injection_dict["luminosity_distance"])
            mag_app = np.interp(t_detect, times, mag_app)

            mag_err = self.error_budget * np.sqrt(np.random.chisquare(df=1, size = len(t_detect)))
            mag_err = np.maximum(mag_err, 0.01)
            mag_err = np.minimum(mag_err, 1)

            self.data[Filter.name] = np.array([t_detect + self.trigger_time, mag_app, mag_err]).T
        
        self.randomize_nondetections()
    
    def create_injection_from_file(self, file, injection_dict):
        with h5py.File(file) as f:
            times = f["times"][:]
            nus = f["nus"][:]
            parameter_names = f["parameter_names"][:].astype(str).tolist()
            test_X_raw = f["test"]["X"][:]

            X = np.array([injection_dict[p] for p in parameter_names])
            
            ind = np.argmin(np.sum( ( (test_X_raw - X)/(np.max(test_X_raw, axis=0) - np.min(test_X_raw, axis=0)) )**2, axis=1))
            X = test_X_raw[ind]

            log_flux = f["test"]["y"][ind]
        
        print(f"Found suitable injection with {dict(zip(parameter_names, X))}")
        mJys = np.exp(log_flux).reshape(len(nus), len(times))
        mJys, times, nus = apply_redshift(mJys, times, nus, injection_dict.get("redshift", 0.0))

        self.data = {}

        for Filter in self.Filters:
            t_detect = self.t_detect[Filter.name]

            mag_abs = Filter.get_mag(mJys, nus)
            mag_app = mag_app_from_mag_abs(mag_abs, injection_dict["luminosity_distance"])
            mag_app = np.interp(t_detect, times, mag_app)

            mag_err = self.error_budget * np.sqrt(np.random.chisquare(df=1, size = len(t_detect)))
            mag_err = np.maximum(mag_err, 0.01)
            mag_err = np.minimum(mag_err, 1)

            self.data[Filter.name] = np.array([t_detect + self.trigger_time, mag_app, mag_err]).T
        
        self.randomize_nondetections()
        return dict(zip(parameter_names, X))


