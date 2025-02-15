from fiesta.inference.lightcurve_model import LightcurveModel, FluxModel

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
                 model: LightcurveModel,
                 data_file: str,
                 filters: list = None,
                 outdir: str = "./benchmarks",
                 metric_name: str = "Linf",
                 ) -> None:
        
        self.model = model
        self.times = self.model.times
        self.file = data_file
        self.outdir = outdir
        
        # Load filters
        if filters is None:
            self.Filters = model.Filters
        else: 
            self.Filters = [Filt for Filt in model.Filters if Filt.name in filters]
        print(f"Loaded filters are: {[Filt.name for Filt in self.Filters]}.")

        # Load metric
        if metric_name == "L2":
            self.metric_name = "$\\mathcal{L}_2$"
            self.metric = lambda y: np.sqrt(trapezoid(x= np.log(self.times) ,y=y**2, axis = -1)) / (np.log(self.times[-1]) - np.log(self.times[0]))
            self.metric2d = lambda y: np.sqrt(trapezoid(x = self.nus, y =trapezoid(x = self.times, y = (y**2).reshape(-1, len(self.nus), len(self.times)) ) ))
            self.file_ending = "L2"
        else:
            self.metric_name = "$\\mathcal{L}_\\inf$"
            self.metric = lambda y: np.max(np.abs(y), axis = -1)
            self.metric2d = lambda y: np.max(np.abs(y), axis = (1,2))
            self.file_ending = "Linf"

        self.get_data()
        self.calculate_error()
        self.get_error_distribution()

    def get_data(self,):
        
        # get the test data
        self.test_mag = {}
        with h5py.File(self.file, "r") as f:
            self.parameter_distributions = ast.literal_eval(f["parameter_distributions"][()].decode('utf-8'))
            self.parameter_names =  f["parameter_names"][:].astype(str).tolist()
            nus = f["nus"][:]

            self.test_X_raw = f["test"]["X"][:]
            test_y_raw = f["test"]["y"][:]
            test_y_raw = test_y_raw.reshape(len(self.test_X_raw), len(f["nus"]), len(f["times"]) ) 
            test_y_raw = interp1d(f["times"][:], test_y_raw, axis = 2)(self.times) # interpolate the test data over the time range of the model
            mJys = np.exp(test_y_raw)

        for Filt in self.Filters:
            self.test_mag[Filt.name] = Filt.get_mags(mJys, nus)
        
        # get the model prediction on the test data
        param_dict = dict(zip(self.parameter_names, self.test_X_raw.T))
        param_dict["luminosity_distance"] = np.ones(len(self.test_X_raw)) * 1e-5
        param_dict["redshift"] = np.zeros(len(self.test_X_raw))
        _, self.pred_mag = self.model.vpredict(param_dict)         
    
    def calculate_error(self,):
        self.error = {}

        for Filt in self.Filters:
            test_y = self.test_mag[Filt.name]
            pred_y = self.pred_mag[Filt.name]
            self.error[Filt.name] = self.metric(test_y - pred_y)

        if isinstance(self.model, FluxModel):
            self.nus = self.model.nus
            log_mJys = np.array([self.model.predict_log_flux(self.test_X_raw[j]) for j in range(len(self.test_X_raw))])
            self.error["total"] = self.metric2d(log_mJys)
        else:
            max_errors = {key: np.max(value) for key, value in self.error.items()}
            max_key = max(max_errors, key=max_errors.get)
            self.error["total"] = self.error[max_key]
    
    def get_error_distribution(self,):
        error_distribution = {}
        for j, p in enumerate(self.parameter_names):
            p_array = self.test_X_raw[:,j]
            bins = np.linspace(self.parameter_distributions[p][0], self.parameter_distributions[p][1], 12)
            # calculate the error histogram with mismatch as weights
            error_distribution[p] = np.histogram(p_array, weights = self.error["total"], bins = bins, density = True)

        self.error_distribution = error_distribution
    
    def benchmark(self,):
        self.print_correlations()
        self.plot_worst_lightcurves()
        self.plot_error_over_time()
        self.plot_error_distribution()

    def plot_lightcurves_mismatch(self,
                                  parameter_labels: list[str] = ["$\\iota$", "$\log_{10}(E_0)$", "$\\theta_c$", "$\log_{10}(n_{\mathrm{ism}})$", "$p$", "$\\epsilon_E$", "$\\epsilon_B$"]
                                  ):
        if self.metric_name == "$\\mathcal{L}_2$":
            vline = self.metric(np.ones(len(self.times)))
            vmin, vmax = 0, vline*2
            bins = np.linspace(vmin, vmax, 25)
        else:
            vline = 1.
            vmin, vmax = 0, 2*vline
            bins = np.linspace(vmin, vmax, 20)
    
        cmap = colors.LinearSegmentedColormap.from_list(name = "mymap", colors = [(0, "lightblue"), (1, "darkred")])
        label_dic = {p: label for p, label in zip(self.parameter_names, parameter_labels)}

        for Filt in self.Filters:

            mismatch = self.error[Filt.name]
            colored_mismatch = cmap(mismatch/vmax)

    
            fig, ax = plt.subplots(len(self.parameter_names)-1, len(self.parameter_names)-1)
            fig.suptitle(f"{Filt.name}: {self.metric_name} norm")
    
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
            outfile  = f"benchmark_{Filt.name}_{self.file_ending}.pdf"
            
            fig.savefig(os.path.join(self.outdir, outfile))
    
    def plot_worst_lightcurves(self,):

        fig, ax = plt.subplots(len(self.Filters) , 1, figsize = (5, 15))
        fig.subplots_adjust(hspace = 0.5, bottom = 0.08, top = 0.98, left = 0.14, right = 0.95)

        for cax, filt in zip(ax, self.Filters):
            ind = np.argmax(self.error[filt.name])
            prediction = self.pred_mag[filt.name][ind]
            cax.plot(self.times, prediction, color = "blue")
            cax.fill_between(self.times, prediction-1, prediction+1, color = "blue", alpha = 0.2)

            cax.plot(self.times, self.test_mag[filt.name][ind], color = "red")
            cax.invert_yaxis()
            cax.set(xlabel = "$t$ in days", ylabel = "mag", xscale = "log", xlim = (self.times[0], self.times[-1]))
            cax.set_title(f"{filt.name}", loc = "right", pad = -20)
            cax.text(0, 0.05, np.array_str(self.test_X_raw[ind], precision = 2), transform = cax.transAxes, fontsize = 7)

        fig.savefig(os.path.join(self.outdir, f"worst_lightcurves_{self.file_ending}.pdf"), dpi = 200)

    def plot_error_over_time(self,):

        fig, ax = plt.subplots(len(self.Filters) , 1, figsize = (5, 15))
        fig.subplots_adjust(hspace = 0.5, bottom = 0.08, top = 0.98, left = 0.14, right = 0.95)

        for cax, filt in zip(ax, self.Filters):
            error = np.abs(self.pred_mag[filt.name] - self.test_mag[filt.name])
            indices = np.linspace(5, len(self.times)-1, 10).astype(int)
            cax.violinplot(error[:, indices], positions=self.times[indices], widths=self.times[indices]/3, points=400)
            cax.set(xlabel = "$t$ in days", ylabel = "error in mag", xscale = "log", xlim = (self.times[0], self.times[-1]), ylim = (0,1.5))
            cax.set_title(f"{filt.name}", loc = "right", pad = -20)
        
        fig.savefig(os.path.join(self.outdir, f"error_over_time.pdf"), dpi = 200)

    def print_correlations(self, ):
        for Filt in self.Filters:
            error = self.error[Filt.name]
            print(f"\n \n \nCorrelations for filter {Filt.name}:\n")
            for j, p in enumerate(self.parameter_names):
                print(f"{p}: {np.corrcoef(self.test_X_raw[:,j], error)[0,1]}")
    
    def plot_error_distribution(self,):

        fig, ax = plt.subplots(len(self.parameter_names), 1, figsize = (4, 18))
        fig.subplots_adjust(hspace = 0.5, bottom = 0.08, top = 0.98, left = 0.09, right = 0.95)
                
        for p, cax in zip(self.parameter_names, ax):
            cax.bar(self.error_distribution[p][1][:-1], self.error_distribution[p][0], width = 1, color = "blue")
            cax.set_xlabel(p)
            cax.set_yticks([])

        fig.savefig(os.path.join(self.outdir, f"error_distribution.pdf"), dpi = 200)
