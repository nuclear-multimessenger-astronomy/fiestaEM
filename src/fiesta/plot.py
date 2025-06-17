from multiprocessing import Value
import matplotlib
import matplotlib.pyplot as plt

pltparams = {"axes.grid": False,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16,
        "figure.constrained_layout.use": True}

plt.rcParams.update(pltparams)

import numpy as np
import pandas as pd

from jaxtyping import Array
from fiesta.logging import logger
from fiesta.inference.lightcurve_model import SurrogateModel
from fiesta.inference.systematic import process_file
from fiesta.inference.likelihood import EMLikelihood


#############################
# DEFAULT SETTINGS / LABELS #
#############################


default_corner_kwargs = dict(bins=40, 
                        smooth=True, 
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        quantiles=[],
                        levels=[0.68, 0.95],
                        plot_density=False, 
                        plot_datapoints=False, 
                        fill_contours=False,
                        max_n_ticks=3, 
                        min_n_ticks=3,
                        save=False,
                        truth_color="darkorange",
                        labelpad=0.2)

latex_labels=dict(inclination_EM="$\\iota$",
                  log10_E0="$\\log_{10}(E_0)$", 
                  thetaCore="$\\theta_{\\mathrm{c}}$", 
                  thetaWing="$\\theta_{\\mathrm{w}}$", 
                  alphaWing="$\\alpha_{\\mathrm{w}}$", 
                  log10_n0="$\\log_{10}(n_{\mathrm{ism}})$",
                  p="$p$", 
                  log10_epsilon_e="$\\log_{10}(\\epsilon_e)$",
                  log10_epsilon_B="$\\log_{10}(\\epsilon_B)$",
                  epsilon_e="$\\epsilon_e$",
                  epsilon_B="$\\epsilon_B$",
                  log10_mej_dyn="$\\log_{10}(m_{\\mathrm{ej,dyn}})$",
                  log10_mej_wind="$\\log_{10}(m_{\\mathrm{ej,wind}})$",
                  v_ej_dyn="$\\bar{v}_{\\mathrm{ej,dyn}}$",
                  v_ej_wind="$\\bar{v}_{\\mathrm{ej,wind}}$",
                  Ye_dyn="$\\bar{Y}_{e,\\mathrm{dyn}}$",
                  Ye_wind="$Y_{e,\\mathrm{wind}}$",
                  luminosity_distance="$d_L$",
                  redshift="$z$",
                  sys_err="$\\sigma_{\mathrm{sys}}$",
                  Gamma0="$\\Gamma_0$")



#######################
#  PLOTTING FUNCTIONS #
#######################


def corner_plot(posterior: dict | pd.DataFrame,
                parameter_names: list[str],
                truths: dict = None,
                color:str = "blue",
                legend_label:str = None,
                fig: matplotlib.figure.Figure = None,
                ax: matplotlib.axes.Axes = None):
    
    try:
        import corner
    except ImportError:
        logger.warning(f"Install corner to create corner plots.")
        return 1, 1
    
    posterior = pd.DataFrame(posterior)
    
    labels= []
    truths_list = []
    for p in parameter_names:
        labels.append(latex_labels.get(p, p))
        truths_list.append(truths.get(p, None))

    if fig is None and ax is None:
        n = len(parameter_names)
        fig, ax = plt.subplots(n, n, figsize = (n*1.5, n*1.5))
    
    if (fig is None and ax is not None) or (fig is not None and ax is None):
        raise ValueError("fig and ax must be either both be specified or both be None.")

    corner.corner(posterior[parameter_names], 
                  fig=fig,
                  color=color,
                  labels=labels,
                  truths=truths_list,
                  **default_corner_kwargs,
                  hist_kwargs=dict(density=True, color=color))
    
    if legend_label is not None:
        
        if len(parameter_names) < 4:
            lx, ly = 0, -1
        else:
            lx, ly = 1, 4

        handle = plt.plot([],[], color=color)[0]
        ax[lx, ly].legend(handles=[handle], labels=[legend_label], fontsize=15, fancybox=False, framealpha=1)
    
    #fig.tight_layout()
    return fig, ax

# TODO: superpose multiple posteriors in one corner plot


class LightcurvePlotter:
    
    def __init__(self, 
                 posterior: dict | pd.DataFrame,
                 likelihood: EMLikelihood,
                 systematics_file: str = None,
                 free_syserr=False):
        
        if systematics_file is not None:
            sys_params_per_filter, t_nodes_per_filter, _= process_file(systematics_file, filters=likelihood.filters)
            likelihood._setup_sys_uncertainty_from_file(sys_params_per_filter, t_nodes_per_filter)
        
        if free_syserr:
            likelihood._setup_sys_uncertainty_free()
        
        self.likelihood = likelihood

        self.tmin = likelihood.tmin
        self.tmax = likelihood.tmax
        self.times_det = likelihood.times_det
        self.mag_det = likelihood.mag_det
        self.mag_err = likelihood.mag_err
        self.times_nondet = likelihood.times_nondet
        self.mag_nondet = likelihood.mag_nondet

        self.model = likelihood.model
        self.posterior = pd.DataFrame(posterior)
        self.fixed_params = likelihood.fixed_params

    def plot_data(self, 
                  ax: matplotlib.axes.Axes, 
                  filt: str,
                  zorder=3,
                  **kwargs):
            
        # Detections
        t, mag, err = self.times_det[filt], self.mag_det[filt], self.mag_err[filt]
        ax.errorbar(t, mag, yerr=err, fmt="o", zorder=zorder, **kwargs)
            
        # Non-detections
        t, mag = self.times_nondet[filt], self.mag_nondet[filt]
        ax.scatter(t, mag, zorder=zorder, marker="v", **kwargs)


    def plot_best_fit_lc(self,
                         ax: matplotlib.axes.Axes,
                         filt: str,
                         zorder=2,
                         **kwargs):

        self._get_best_fit_lc()
        ax.plot(self.t_best_fit, self.best_fit_lc[filt], zorder=zorder, **kwargs)
        
    
    def _get_best_fit_lc(self,):

        if hasattr(self, "_best_fit_lc_determined"):
            return

        best_ind = np.argmax(self.posterior["log_prob"])
        self.best_fit_params = {}
        for key in self.posterior.keys():
            self.best_fit_params[key] = self.posterior[key][best_ind]
        
        self.best_fit_params.update(self.fixed_params)

        t, model_mag = self.model.predict(self.best_fit_params)
        mask = (t>=self.tmin) & (t<=self.tmax)

        self.t_best_fit = t[mask]
        self.best_fit_lc = {}
        for filt in model_mag.keys():
            self.best_fit_lc[filt] = model_mag[filt][mask]
        self._best_fit_lc_determined = True

    def plot_sample_lc(self,
                       ax: matplotlib.axes.Axes,
                       filt: str,
                       zorder=1):
        
        self._get_samples_lcs()

        for j in range(200):
            ax.plot(self.t_sample_lc[j], self.sample_lc[filt][j], color="grey", alpha=0.05, zorder=zorder, rasterized=True)
    
    def _get_samples_lcs(self,):
        
        if hasattr(self, "_sample_lcs_determined"):
            return

        total_nb_samples = self.posterior.values.shape[0]
        ind = np.random.choice(total_nb_samples, 200, replace=False)

        params = {}
        for key in self.posterior.keys():
            params[key] = self.posterior[key][ind]
        for key in self.fixed_params:
            params[key] = np.ones(200) * self.fixed_params[key]
        
        self.t_sample_lc, self.sample_lc = self.model.vpredict(params)
        self._sample_lcs_determined = True

    def plot_sys_uncertainty_band(self,
                                  ax: matplotlib.axes.Axes,
                                  filt: str,
                                  systematics_file: str,
                                  zorder=2,
                                  **kwargs):
        self._get_best_fit_lc()

        sys_params_per_filter, t_nodes_per_filter, _ = process_file(systematics_file, [filt])
        sys_params_per_filter = sys_params_per_filter[filt]
        t_nodes_per_filter = t_nodes_per_filter[filt]

        if t_nodes_per_filter is None:
            t_nodes_per_filter = np.linspace(self.tmin, self.tmax, len(sys_params_per_filter))
        
        
        sys_param_array = np.array([self.best_fit_params[p] for p in sys_params_per_filter])
        sigma_sys = np.interp(self.t_best_fit, t_nodes_per_filter, sys_param_array)

        ax.fill_between(self.t_best_fit, self.best_fit_lc[filt] + sigma_sys, self.best_fit_lc[filt] - sigma_sys, alpha=0.1, zorder=zorder, **kwargs)

        
                                  
