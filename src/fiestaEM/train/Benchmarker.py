from fiestaEM.inference.lightcurve_model import AfterglowpyLightcurvemodel
import afterglowpy as grb
from fiestaEM.constants import days_to_seconds
from fiestaEM import conversions
from fiestaEM import utils
from fiestaEM.utils import Filter

from jaxtyping import Array, Float

import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable

from scipy.integrate import trapezoid

# TODO: get a benchmarker class for all surrogate model
class Benchmarker:

    name: str
    model_dir: str
    filters: list[Filter]
    n_test_data: int
    metric_name: str
    jet_type: int
    model: AfterglowpyLightcurvemodel

    def __init__(self,
                 name: str,
                 model_dir: str,
                 filters: list[str],
                 n_test_data: int = 3000,
                 remake_test_data: bool = False,
                 metric_name: str = "$\\mathcal{L}_\\inf$",
                 jet_type = -1
                 ) -> None:
        
        self.name = name
        self.model_dir = model_dir
        self.load_filters(filters)
        self.model = AfterglowpyLightcurvemodel(name = self.name,
                                                directory = self.model_dir,
                                                filters = filters)
        self.times = self.model.times
        self._times_afterglowpy = self.times * days_to_seconds
        self.jet_type = jet_type

        self.parameter_names = self.model.metadata["parameter_names"]
        train_X_raw = np.load(self.model_dir+"raw_data_training.npz")["X_raw"]
        self.parameter_boundaries = np.array([np.min(train_X_raw, axis = 0), np.max(train_X_raw, axis = 0)])
        
        if os.path.exists(self.model_dir+"/raw_data_test.npz") and not remake_test_data:
            self.load_test_data()
        else:
           self.get_test_data(n_test_data)

        self.metric_name = metric_name
        mask = np.logical_and(self.times>8, self.times<800)
        if metric_name == "$\\mathcal{L}_2$":
            self.metric = lambda y: np.sqrt(trapezoid(x= self.times[mask],y=y[mask]**2))
        else:
            self.metric = lambda y: np.max(np.abs(y[mask]))

    def __repr__(self) -> str:
        return f"Surrogate_Benchmarker(name={self.name}, model_dir={self.model_dir})"

    def load_filters(self, filters: list[str]):
        self.filters = []
        for filter in filters:
            try:
                self.filters.append(utils.Filter(filter))
            except:
                raise Exception(f"Filter {filter} not available.")
        
    def get_test_data(self, n_test_data):
        test_X_raw = np.empty((n_test_data, len(self.parameter_names)))
        test_y_raw = {filter.name: np.empty((n_test_data, len(self.times))) for filter in self.filters}
        prediction_y_raw = {filter.name: np.empty((n_test_data, len(self.times))) for filter in self.filters}

        print(f"Determining test data for {n_test_data} random points within parameter grid.")
        for j in tqdm.tqdm(range(n_test_data)):
            test_X_raw[j] = np.random.uniform(low = self.parameter_boundaries[0], high = self.parameter_boundaries[1])
            param_dict = {name: x for name, x in zip(self.parameter_names, test_X_raw[j])}

            prediction = self.model.predict(param_dict)

            for filt in self.filters:
                param_dict["nu"] = filt.nu
                prediction_y_raw[filt.name][j] = prediction[filt.name]
                mJys = self._call_afterglowpy(param_dict)
                test_y_raw[filt.name][j] = conversions.mJys_to_mag_np(mJys)
                
        self.test_X_raw = test_X_raw
        self.test_y_raw = test_y_raw
        self.prediction_y_raw = prediction_y_raw
        self.n_test_data = n_test_data
        
        #for saving
        test_saver = {"test_"+key: test_y_raw[key] for key in test_y_raw.keys()}
        np.savez(os.path.join(self.model_dir, "raw_data_test.npz"), X = test_X_raw, **test_saver)
    
    def load_test_data(self, ):
        
        test_data = np.load(self.model_dir+"/raw_data_test.npz")
        self.test_X_raw = test_data["X"]
        self.test_y_raw = {filt.name: test_data["test_"+filt.name] for filt in self.filters}
        self.n_test_data = len(self.test_X_raw)

        self.prediction_y_raw = {filt.name: np.empty((self.n_test_data, len(self.times))) for filt in self.filters}
        for j, X in enumerate(self.test_X_raw):
            param_dict = {name: x for name, x in zip(self.parameter_names, X)}
            prediction = self.model.predict(param_dict)
            for filt in self.filters:
                self.prediction_y_raw[filt.name][j] = prediction[filt.name]

    def _call_afterglowpy(self,
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
        Z["d_L"]       = 3.086e19 # fix at 10 pc, so that AB magnitude equals absolute magnitude
        if "inclination_EM" in list(params_dict.keys()):
            Z["thetaObs"]  = params_dict["inclination_EM"]
        else:
            Z["thetaObs"]  = params_dict["thetaObs"]
        if self.jet_type == 1 or self.jet_type == 4:
            Z["b"] = params_dict["b"]
        if "thetaWing" in list(params_dict.keys()):
            Z["thetaWing"] = params_dict["thetaWing"]
        
        # Afterglowpy returns flux in mJys
        mJys = grb.fluxDensity(self._times_afterglowpy, params_dict["nu"], **Z)
        return mJys
    
    def calculate_mismatch(self, filter):
        mismatch = np.empty(self.n_test_data)

        for j in range(self.n_test_data):
            mismatch[j] = self.metric(self.prediction_y_raw[filter][j] - self.test_y_raw[filter][j])

        return mismatch



    def plot_lightcurves_mismatch(self,
                                  filter: str,
                                  parameter_labels: list[str] = ["$\\iota$", "$\log_{10}(E_0)$", "$\\theta_c$", "$\log_{10}(n_{\mathrm{ism}})$", "$p$", "$\\epsilon_E$", "$\\epsilon_B$"]
                                  ):
        if self.metric_name == "$\\mathcal{L}_2$":
            bins = np.arange(0, 100, 5)
            vmin, vmax = 0, 50
            vline = np.sqrt(trapezoid(x = self.times, y = np.ones(len(self.times))))
        else:
            bins = np.arange(0, 3, 0.5)
            vmin, vmax = 0, 3
            vline = 1.
        
        mismatch = self.calculate_mismatch(filter)
        
        cmap = colors.LinearSegmentedColormap.from_list(name = "mymap", colors = [(0, "lightblue"), (1, "darkred")])
        colored_mismatch = cmap(mismatch/vmax)

        label_dic = {p: label for p, label in zip(self.parameter_names, parameter_labels)}

        fig, ax = plt.subplots(len(self.parameter_names)-1, len(self.parameter_names)-1)
        fig.suptitle(f"{filter}: {self.metric_name} norm")

        for j, p in enumerate(self.parameter_names[1:]):
            for k, pp in enumerate(self.parameter_names[:j+1]):
                sort = np.argsort(mismatch)

                ax[j,k].scatter(self.test_X_raw[sort,k], self.test_X_raw[sort,j+1], c = colored_mismatch[sort], s = 1)

                ax[j,k].set_xlim((self.test_X_raw[:,k].min(), self.test_X_raw[:,k].max()))
                ax[j,k].set_ylim((self.test_X_raw[:,j+1].min(), self.test_X_raw[:,j+1].max()))
            

                if k!=0:
                    ax[j,k].set_yticklabels([])

                if j!=len(self.parameter_names)-2:
                    ax[j,k].set_xticklabels([])

                ax[-1,k].set_xlabel(label_dic[pp])
            ax[j,0].set_ylabel(label_dic[p])
                
            for cax in ax[j, j+1:]:
                cax.set_axis_off()
        
        ax[0,-1].set_axis_on()
        ax[0,-1].hist(mismatch, density = True, histtype = "step", bins = bins,)
        ax[0,-1].vlines([vline], *ax[0,-1].get_ylim(), colors = ["lightgrey"], linestyles = "dashed")
        ax[0,-1].set_yticks([])
            
        fig.colorbar(ScalarMappable(norm=colors.Normalize(vmin = vmin, vmax = vmax), cmap = cmap), ax = ax[1:-1, -1])
        return fig, ax

    def print_correlations(self,
                           filter: str,):
        
        mismatch = self.calculate_mismatch(filter)

        
        print(f"\n \n \nCorrelations for filter {filter}:\n")
        corrcoeff = []
        for j, p in enumerate(self.parameter_names):
            print(f"{p}: {np.corrcoef(self.test_X_raw[:,j], mismatch)[0,1]}")
    
    def plot_worst_lightcurve(self,filter):
        
        mismatch = self.calculate_mismatch(filter)
        ind = np.argsort(mismatch)[-1]
        
        fig, ax = plt.subplots(1,1)
        fig.suptitle(f"{filter}")
        ax.plot(self.times, self.prediction_y_raw[filter][ind], color = "blue")
        ax.fill_between(self.times, self.prediction_y_raw[filter][ind]-1, self.prediction_y_raw[filter][ind]+1, color = "blue", alpha = 0.2)
        ax.plot(self.times, self.test_y_raw[filter][ind], color = "red")
        plt.gca().invert_yaxis()
        ax.set(xlabel = "$t$ in days", ylabel = "mag")
        
        return fig, ax
