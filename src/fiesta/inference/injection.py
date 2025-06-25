"""Functions for creating and handling injections"""
import argparse
import os

import h5py
from jaxtyping import Float, Array
import numpy as np

from fiesta.inference.lightcurve_model import LightcurveModel
from fiesta.conversions import mag_app_from_mag_abs, apply_redshift
from fiesta.filters import Filter
from fiesta.utils import write_event_data
from fiesta.logging import logger

from fiesta.train.AfterglowData import RunAfterglowpy, RunPyblastafterglow

# TODO: get the parser going
def get_parser(**kwargs):
    add_help = kwargs.get("add_help", True)

    parser = argparse.ArgumentParser(
        description="Inference on kilonova and GRB parameters.",
        add_help=add_help,
    )


class InjectionBase:
    """
    Base class to create synthetic injection lightcurves.
    The injection model is first initialized with the following parameters:
        filters (list): List of filters in which the synthetic data should be given out.
        tmin (float): Time of earliest synthetic detection possible in days. Defaults to 0.1.
        tmax (float): Time of latest synthetic detection possible in days. Defaults to 10.0
        N_datapoints (int): Total number of datapoints (across all filters) for the synthetic lightcurve. Defaults to 10.
        t_detect (dict[str, Array]): Detection time points in each filter. If none is specified, then the detection times will be sampled randomly.
        error_budget (float): Typical measurement error scale of the synthetic data. Defaults to 1.
        detection_limit (float): Synthetic datapoints with mangnitude higher than this value (i.e. less brighter) will be turned into nondetections. Defaults to np.inf.
        nondetections (bool): Additional to detection_limit, this turns some of the synthetic datapoints to nondetections. Defaults to False.
        nondetections_fraction: If nondetections is True, then this will determine the fractions of N_datapoints turned into nondetections. Defaults to 0.1.

    Then one can call the .create_injection() method to get synthetic lightcurve data. 
    The method .write_to_file() writes the synthetic lightcurve data to file.    
    """

    def __init__(self,
                 filters: list[str],
                 trigger_time: float,
                 tmin: Float = 0.1,
                 tmax: Float = 10.0,
                 N_datapoints: int = 10,
                 t_detect: dict[str, Array] = None,
                 error_budget: Float = 1.0,
                 nondetections: bool = False,
                 nondetections_fraction: Float = 0.1,
                 detection_limit: Float = np.inf):
        
        self.Filters = [Filter(filt) for filt in filters]
        logger.info(f"Creating injection with filters: {filters}")
        self.trigger_time = trigger_time
        self.tmin = tmin
        self.tmax = tmax

        if t_detect is not None:
           self.t_detect = t_detect
        else:
            self.create_t_detect(tmin, tmax, N_datapoints)

        self.error_budget = error_budget
        self.nondetections = nondetections
        self.nondetections_fraction = nondetections_fraction
        self.detection_limit = detection_limit
    
    def create_t_detect(self, tmin, tmax, N):
        """Create a time grid for the injection data."""

        self.t_detect = {}
        points_list = np.random.multinomial(N, [1/len(self.Filters)]*len(self.Filters)) # random number of time points in each filter

        for points, Filt in zip(points_list, self.Filters):
            t = np.exp(np.random.uniform(np.log(tmin), np.log(tmax), size=points))
            t = np.sort(t)
            t[::2] *= np.random.uniform(1, (tmax/tmin)**(1/points), size = len(t[::2])) # correlate the time points
            t[::3] *= np.random.uniform(1, (tmax/tmin)**(1/points), size = len(t[::3])) # correlate the time points
            mask = (t<tmin) | (t>tmax)
            t[mask] = np.exp(np.random.uniform(np.log(tmin), np.log(tmax), size=np.sum(mask)))
            self.t_detect[Filt.name] = np.sort(t)
    
    def create_injection(self,
                         injection_dict: dict[str, Float],
                         file: str = None):
        

        if file is None:
            times, mag_app = self._get_injection_lc(injection_dict)
        else:
            times, mag_app, injection_dict = self._get_injection_lc_from_file(injection_dict, file)
        
        self.injection_dict = injection_dict
        self.data = {}

        for Filter in self.Filters:
            t_detect = self.t_detect[Filter.name]
            mu = np.interp(t_detect, times, mag_app[Filter.name])

            sigma = self.error_budget * np.sqrt(np.random.chisquare(df=1, size = len(t_detect)))
            sigma = np.maximum(sigma, 0.01)
            sigma = np.minimum(sigma, 1)

            mag_measured = np.random.normal(loc=mu, scale=sigma)
            
            # apply detection limit
            not_detected = np.where(mag_measured > self.detection_limit)
            mag_measured[not_detected] = self.detection_limit
            sigma[not_detected] = np.inf

            self.data[Filter.name] = np.array([t_detect + self.trigger_time, mag_measured, sigma]).T
        
        # add additional non detections
        self.randomize_nondetections()
    
    def _get_injection_lc_from_file(self, injection_dict, file):
        """Create a synthetic lightcurve from training data file given the parameters in injection_dict."""
        with h5py.File(file) as f:
            times = f["times"][:]
            nus = f["nus"][:]
            parameter_names = f["parameter_names"][:].astype(str).tolist()
            test_X_raw = f["test"]["X"][:]

            X = np.array([injection_dict[p] for p in parameter_names])
            ind = np.argmin(np.sum( ( (test_X_raw - X)/(np.max(test_X_raw, axis=0) - np.min(test_X_raw, axis=0)) )**2, axis=1))
            X = test_X_raw[ind]

            log_flux = f["test"]["y"][ind]
        
        injection_dict.update(dict(zip(parameter_names, X)))
        injection_dict["redshift"] = injection_dict.get("redshift", 0.0)
        print(f"Found suitable injection with {injection_dict}")
        mJys = np.exp(log_flux).reshape(len(nus), len(times))
        mJys, times_obs, nus = apply_redshift(mJys, times, nus, injection_dict["redshift"])

        if self.tmin < times_obs[0] or self.tmax > times_obs[-1]:
            raise ValueError(f"Time range {(self.tmin, self.tmax)} is too large for file {file} with time range {(times[0], times[-1])} at redshift {injection_dict['redshift']}.")

        mags = {}
        for Filter in self.Filters:
            mag_abs = Filter.get_mag(mJys, nus)
            mags[Filter.name] = mag_app_from_mag_abs(mag_abs, injection_dict["luminosity_distance"])

        return times_obs, mags, injection_dict
    
    def randomize_nondetections(self,):
        if not self.nondetections:
            return
        
        N = np.sum([len(self.t_detect[Filt.name]) for Filt in self.Filters])
        nondets_list = np.random.multinomial(int(N*self.nondetections_fraction), [1/len(self.Filters)]*len(self.Filters)) # random number of non detections in each filter

        for nondets, Filt in zip(nondets_list, self.Filters):
            inds = np.random.choice(np.arange(len(self.data[Filt.name])), size=nondets, replace=False)
            self.data[Filt.name][inds] += np.array([0, -5., np.inf])
    
    def write_to_file(self, file: str):
        write_event_data(file, self.data)
        dir = os.path.dirname(file)
        with open(os.path.join(dir,"param_dict.dat"), "w") as o:
             o.write(str(self.injection_dict))

class InjectionSurrogate(InjectionBase):
    
    def __init__(self, 
                 model: LightcurveModel,
                 *args,
                 **kwargs):
        
        self.model = model
        super().__init__(*args, **kwargs)
        
    def _get_injection_lc(self, injection_dict):
        """Create a synthetic lightcurve from a surrogate given the parameters in injection_dict."""

        injection_dict["luminosity_distance"] = injection_dict.get('luminosity_distance', 1e-5)
        injection_dict["redshift"] = injection_dict.get('redshift', 0)

        times, mags = self.model.predict(injection_dict)
        
        if self.tmin < times[0] or self.tmax > times[-1]:
            raise ValueError(f"Time range {(self.tmin, self.tmax)} is too large for model {self.model} with time range {(self.model.times[0], self.model.times[-1])} at redshift {injection_dict['redshift']}.")
        return times, mags

class InjectionAfterglowpy(InjectionBase):
    
    def __init__(self,
                 jet_type: int = -1,
                 *args,
                 **kwargs):
        
        self.jet_type = jet_type
        super().__init__(*args, **kwargs)
        
    def _get_injection_lc(self, injection_dict):
        """Create a synthetic lightcurve from afterglowpy given the parameters in injection_dict."""

        nus = [nu for Filter in self.Filters for nu in Filter.nus]
        times = [t for Filter in self.Filters for t in self.t_detect[Filter.name]]

        nus = np.sort(nus)
        times = np.sort(times)

        afgpy = RunAfterglowpy(self.jet_type, times, nus, [list(injection_dict.values())], injection_dict.keys())
        _, log_flux = afgpy(0)
        mJys  = np.exp(log_flux).reshape(len(nus), len(times))

        mags = {}
        for Filter in self.Filters:
            mag_abs = Filter.get_mag(mJys, nus) # even when 'luminosity_distance' is passed to RunAfterglowpy, it will return the abs mag (with redshift)
            mags[Filter.name] = mag_app_from_mag_abs(mag_abs, injection_dict["luminosity_distance"])

        return times, mags

class InjectionPyblastafterglow(InjectionBase):
    
    def __init__(self,
                 jet_type: str = "tophat",
                 *args,
                 **kwargs):
        
        self.jet_type = jet_type
        super().__init__(*args, **kwargs)
        
    def _get_injection_lc(self, injection_dict):
        """Create a synthetic lightcurve from pyblastafterglow given the parameters in injection_dict."""

        nus = [nu for Filter in self.Filters for nu in Filter.nus]
        times = [t for Filter in self.Filters for t in self.t_detect[Filter.name]]

        nus = np.sort(nus)
        times = np.sort(times)
        nus = np.logspace(np.log10(nus[0]), np.log10(nus[-1]), 128) #pbag only takes log (or linear) spaced arrays
        times = np.logspace(np.log10(times[0]), np.log10(times[-1]), 100)

        pbag = RunPyblastafterglow(self.jet_type, times, nus, [list(injection_dict.values())], injection_dict.keys())
        _, log_flux = pbag(0)
        mJys  = np.exp(log_flux).reshape(len(nus), len(times))

        mags = []
        for Filter in self.Filters:
            mags[Filter.name] = Filter.get_mag(mJys, nus)
        
        return times, mags

class InjectionKN(InjectionBase):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
    
    def _get_injection_lc(self, injection_dict):
        raise NotImplementedError(f"No direct calculation for KN injection available, use a training data file instead.")