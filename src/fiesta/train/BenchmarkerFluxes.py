from fiesta.inference.lightcurve_model import LightcurveModel
import afterglowpy as grb
from fiesta.constants import days_to_seconds, c
from fiesta import conversions
from fiesta import utils

from jaxtyping import Array, Float, Int

import tqdm
import os
import ast
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable

from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

class Benchmarker:

    def __init__(self,
                 name: str,
                 model_dir: str,
                 filters: list[str],
                 MODEL = LightcurveModel,
                 metric_name: str = "$\\mathcal{L}_\\inf$"
                 ) -> None:
        
        self.name = name
        self.model_dir = model_dir
        self.model = MODEL(name = self.name,
                           directory = self.model_dir,
                           filters = filters)

        self.times = self.model.times
        self.nus = self.model.metadata["nus"]
        self.load_filters(filters)

        self.get_test_data()
        self.lightcurve_test_data()


        self.metric_name = metric_name
        if metric_name == "$\\mathcal{L}_2$":
            self.metric = lambda y: np.sqrt(trapezoid(x= self.times,y=y**2))
            self.metric2d = lambda y: np.sqrt(trapezoid(x = self.nus, y =trapezoid(x = self.times, y = (y**2).reshape(len(self.nus), len(self.times)) ) ))
        else:
            self.metric = lambda y: np.max(np.abs(y))
            self.metric2d = self.metric
        
        self.calculate_mismatch()
        self.get_error_distribution()
        
    def __repr__(self) -> str:
        return f"Surrogte_Benchmarker(name={self.name}, model_dir={self.model_dir})"

    def load_filters(self, filters: list[str]):
        self.filters = []
        for filter in filters:
            try:
                self.filters.append(utils.Filter(filter))
            except:
                raise Exception(f"Filter {filter} not available.")
    
    def get_test_data(self,):

        file = [f for f in os.listdir(self.model_dir) if f.endswith("_raw_data.h5")][0]

        with h5py.File(os.path.join(self.model_dir, file), "r") as f:
            self.parameter_distributions = ast.literal_eval(f["parameter_distributions"][()].decode('utf-8'))
            self.parameter_names =  f["parameter_names"][:].astype(str).tolist()
            self.test_X_raw = f["test"]["X"][:]
            y_raw = f["test"]["y"][:]
            y_raw = y_raw.reshape(len(self.test_X_raw), len(f["nus"]), len(f["times"]) ) 
            y_raw = interp1d(f["times"][:], y_raw, axis = 2)(self.times) # interpolate the test data over the time range of the model
            y_raw = interp1d(f["nus"][:], y_raw, axis = 1)(self.nus) # interpolate the test data over the frequency range of the model
            self.fluxes_raw = y_raw.reshape(len(self.test_X_raw), len(self.nus) * len(self.times) )
    
    def lightcurve_test_data(self, ):
        
        self.n_test_data = len(self.test_X_raw)
        self.prediction_y_raw = {filt.name: np.empty((self.n_test_data, len(self.times))) for filt in self.filters}
        self.test_y_raw = {filt.name: np.empty((self.n_test_data, len(self.times))) for filt in self.filters}
        self.prediction_log_fluxes = np.empty((self.n_test_data, len(self.nus) * len(self.times)))
        for j, X in enumerate(self.test_X_raw):
            param_dict = {name: x for name, x in zip(self.parameter_names, X)}
            prediction = self.model.predict(param_dict)
            self.prediction_log_fluxes[j] = self.model.predict_log_flux(X)
            for filt in self.filters:
                self.prediction_y_raw[filt.name][j] = prediction[filt.name]
                self.test_y_raw[filt.name][j] = self.convert_to_mag(filt.nu, self.fluxes_raw[j])
    
    def convert_to_mag(self, nu, flux):
        flux = flux.reshape(len(self.model.metadata["nus"]), len(self.model.times))
        flux = np.exp(flux)
        flux = np.array([np.interp(nu, self.model.metadata["nus"], column) for column in flux.T]) 
        mag = -48.6 + -1 * np.log10(flux*1e-3 / 1e23) * 2.5
        return mag

    
    ### Diagnostics ###
   
    def calculate_mismatch(self):
        mismatch = {}
        for filt in self.filters:
            array = np.empty(self.n_test_data)    
            for j in range(self.n_test_data):
                array[j] = self.metric(self.prediction_y_raw[filt.name][j] - self.test_y_raw[filt.name][j])
            mismatch[filt.name] = array
        
        array = np.empty(self.n_test_data)
        for j in range(self.n_test_data):
            array[j] = self.metric2d(self.prediction_log_fluxes[j]-self.fluxes_raw[j])
        
        mismatch["total"] = array
        self.mismatch = mismatch

    def plot_lightcurves_mismatch(self,
                                  filter: str,
                                  parameter_labels: list[str] = ["$\\iota$", "$\log_{10}(E_0)$", "$\\theta_c$", "$\log_{10}(n_{\mathrm{ism}})$", "$p$", "$\\epsilon_E$", "$\\epsilon_B$"]
                                  ):
        if self.metric_name == "$\\mathcal{L}_2$":
            vline = np.sqrt(trapezoid(x = self.times, y = 0.2*np.ones(len(self.times))))
            vmin, vmax = 0, vline*2
            bins = np.linspace(vmin, vmax, 25)
        else:
            vline = 1.
            vmin, vmax = 0, 2*vline
            bins = np.linspace(vmin, vmax, 20)
        
        mismatch = self.mismatch[filter]
        
        cmap = colors.LinearSegmentedColormap.from_list(name = "mymap", colors = [(0, "lightblue"), (1, "darkred")])
        colored_mismatch = cmap(mismatch/vmax)

        label_dic = {p: label for p, label in zip(self.parameter_names, parameter_labels)}

        fig, ax = plt.subplots(len(self.parameter_names)-1, len(self.parameter_names)-1)
        fig.suptitle(f"{filter}: {self.metric_name} norm")

        for j, p in enumerate(self.parameter_names[1:]):
            for k, pp in enumerate(self.parameter_names[:j+1]):
                sort = np.argsort(mismatch)

                ax[j,k].scatter(self.test_X_raw[sort,k], self.test_X_raw[sort,j+1], c = colored_mismatch[sort], s = 1, rasterized = True)

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
        mismatch = self.mismatch[filter]
        print(f"\n \n \nCorrelations for filter {filter}:\n")
        corrcoeff = []
        for j, p in enumerate(self.parameter_names):
            print(f"{p}: {np.corrcoef(self.test_X_raw[:,j], mismatch)[0,1]}")

    def get_error_distribution(self):
        error_distribution = {}
        for j, p in enumerate(self.parameter_names):
            p_array = self.test_X_raw[:,j]
            #bins = (np.array(self.parameter_distributions[p][:-1]) + np.array(self.parameter_grid[p][1:]))/2
            #bins = [self.parameter_grid[p][0] ,*bins, self.parameter_grid[p][-1]]
            bins = np.linspace(self.parameter_distributions[p][0], self.parameter_distributions[p][1], 12)
            # calculate the error histogram with mismatch as weights
            error_distribution[p], _ = np.histogram(p_array, weights = self.mismatch["total"], bins = bins, density = True)
            error_distribution[p] = error_distribution[p]/np.sum(error_distribution[p])

        self.error_distribution = error_distribution


    def plot_worst_lightcurves(self,):

        fig, ax = plt.subplots(len(self.filters) , 1, figsize = (5, 15))
        fig.subplots_adjust(hspace = 0.5, bottom = 0.08, top = 0.98, left = 0.14, right = 0.95)

        for cax, filt in zip(ax, self.filters):
            ind = np.argmax(self.mismatch[filt.name])
            prediction = self.prediction_y_raw[filt.name][ind]
            cax.plot(self.times, prediction, color = "blue")
            cax.fill_between(self.times, prediction-1, prediction+1, color = "blue", alpha = 0.2)
            cax.plot(self.times, self.test_y_raw[filt.name][ind], color = "red")
            cax.invert_yaxis()
            cax.set(xlabel = "$t$ in days", ylabel = "mag", xscale = "log", xlim = (self.times[0], self.times[-1]))
            cax.set_title(f"{filt.name}", loc = "right", pad = -20)
            cax.text(0, 0.05, np.array_str(self.test_X_raw[ind], precision = 2), transform = cax.transAxes, fontsize = 7)

        return fig, ax

    def plot_error_over_time(self,):

        fig, ax = plt.subplots(len(self.filters) , 1, figsize = (5, 15))
        fig.subplots_adjust(hspace = 0.5, bottom = 0.08, top = 0.98, left = 0.14, right = 0.95)

        for cax, filt in zip(ax, self.filters):
            error = np.abs(self.prediction_y_raw[filt.name] - self.test_y_raw[filt.name])
            indices = np.linspace(5, len(self.times)-1, 10).astype(int)
            cax.violinplot(error[:, indices], positions = self.times[indices], widths = self.times[indices]/3)
            cax.set(xlabel = "$t$ in days", ylabel = "error in mag", xscale = "log", xlim = (self.times[0], self.times[-1]), ylim = (0,1.5))
            cax.set_title(f"{filt.name}", loc = "right", pad = -20)
        return fig, ax
    
    def plot_error_distribution(self,):
        mismatch = self.mismatch["total"]

        fig, ax = plt.subplots(len(self.parameter_names), 1, figsize = (4, 18))
        fig.subplots_adjust(hspace = 0.5, bottom = 0.08, top = 0.98, left = 0.09, right = 0.95)
                
        for j, cax in enumerate(ax):
            p_array = self.test_X_raw[:,j]
            p = self.parameter_names[j]
            bins = np.linspace(self.parameter_distributions[p][0], self.parameter_distributions[p][1], 12)

            cax.hist(p_array, weights = self.mismatch["total"], color = "blue", bins = bins, density = True, histtype = "step")
            cax.set_xlabel(self.parameter_names[j])
            cax.set_yticks([])

        return fig, ax
        



