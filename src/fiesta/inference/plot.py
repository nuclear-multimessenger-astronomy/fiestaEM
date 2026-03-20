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
        "figure.constrained_layout.use": False}

plt.rcParams.update(pltparams)

import numpy as np
import pandas as pd
from jaxtyping import Array
import jax
import jax.numpy as jnp

from fiesta.logging import logger
from fiesta.inference.systematic import process_file
from fiesta.inference.likelihood import LikelihoodBase


#############################
# DEFAULT SETTINGS / LABELS #
#############################


default_corner_kwargs = dict(bins=40, 
                        smooth=True, 
                        label_kwargs=dict(fontsize=14),
                        title_kwargs=dict(fontsize=14), 
                        quantiles=[],
                        levels=[0.68, 0.95],
                        plot_density=False, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=3,
                        min_n_ticks=3,
                        save=False,
                        truth_color="darkorange",
                        labelpad=0.2)

latex_labels=dict(inclination_EM="$\\iota$ [rad]",
                  log10_E0="$\\log_{10}(E_0)$ [erg]", 
                  thetaCore="$\\theta_{\\mathrm{c}}$ [rad]", 
                  thetaWing="$\\theta_{\\mathrm{w}}$ [rad]", 
                  alphaWing="$\\alpha_{\\mathrm{w}}$", 
                  log10_n0="$\\log_{10}(n_{\\mathrm{ism}})$ [cm$^{-3}$]",
                  p="$p$", 
                  log10_epsilon_e="$\\log_{10}(\\epsilon_e)$",
                  log10_epsilon_B="$\\log_{10}(\\epsilon_B)$",
                  epsilon_e="$\\epsilon_e$",
                  epsilon_B="$\\epsilon_B$",
                  log10_mej_dyn="$\\log_{10}(m_{\\mathrm{ej,dyn}})$ [$M_\\odot$]",
                  log10_mej_wind="$\\log_{10}(m_{\\mathrm{ej,wind}})$ [$M_\\odot$]",
                  v_ej_dyn="$\\bar{v}_{\\mathrm{ej,dyn}}$ [$c$]",
                  v_ej_wind="$\\bar{v}_{\\mathrm{ej,wind}}$ [$c$]",
                  Ye_dyn="$\\bar{Y}_{e,\\mathrm{dyn}}$",
                  Ye_wind="$Y_{e,\\mathrm{wind}}$",
                  luminosity_distance="$d_L$ [Mpc]",
                  redshift="$z$",
                  sys_err="$\\sigma_{\\mathrm{sys}}$ [mag]",
                  Gamma0="$\\Gamma_0$",
                  timeshift = "$t_0$ [days]",
                  log10_Menv = "$\\log_{10}(M_{\\rm e})$ [$M_\\odot$]",
                  log10_Renv = "$\\log_{10}(R_{\\rm e})$ [cm]",
                  log10_Ee = "$\\log_{10}(E_e)$ [erg]", 
                  em_syserr = "$\\sigma_{\\rm sys}$ [mag]",
                  Ebv = "$E(B-V)$ [mag]",
                  amplitude = "$A$",
                  supernova_mag_stretch = "$s$")



#######################
#  PLOTTING FUNCTIONS #
#######################


def corner_plot(posterior: dict | pd.DataFrame,
                parameter_names: list[str],
                truths: dict | None = None,
                color: str = "blue",
                legend_label: str | None = None,
                fig: matplotlib.figure.Figure | None = None,
                ax: matplotlib.axes.Axes | None = None,
                **kwargs):
    """
    Make a nice corner plot from the posterior with automated parameter labels.

    Args:
       posterior (dict | pd.DataFrame): posterior samples for which to do the corner plot.
       parameter_names (list[str]): parameters from posterior that should be included in the corner plot.
       truths (dict[str, float] | None): True (injected values) for some of the parameters. Defaults to None.
       color (str): color for the corner plot contours. Defaults to blue.
       legend_label (str): Label for the legend. If not set, no legend will be shown. Defaults to None.
       fig (matplotlib.figure.Figure): Figure over which to do the corner plot. If set, ax must also be provided. Defaults to None.
       ax (matplotlib.axes.Axes): Axes over which to do the corner plot. If set, fig must also be provided. Defaults to None.

    Returns:
        fig (matplotlib.figure.Figure): Figure with the corner plot.
        ax (matplotlib.axes.Axes): array of axes
    """
    
    try:
        import corner
    except ImportError:
        logger.warning("Install corner to create corner plots.")
        return None, None

    if truths is None:
        truths = {}

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

    corner_args = default_corner_kwargs.copy()
    corner_args.update(kwargs)

    corner.corner(posterior[parameter_names], 
                  fig=fig,
                  color=color,
                  labels=labels,
                  truths=truths_list,
                  **corner_args,
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
    """
    Interface to plot lightcurves from a given posterior.

    Args:
        posterior (dict | pd.DataFrame): Posterior samples for which the light curves should be plotted.
        likelihood (EMLikelihood): Likelihood object that was used to sample the posterior.
        systematics_file (str): Systematics file that was used to sample the posterior. Defaults to None.
        free_syserr (bool): Whether a global systematic uncertainty was sampled freely. Defaults to False. Will overwrite systematics_file.
    """
    
    def __init__(self, 
                 posterior: dict | pd.DataFrame,
                 likelihood: LikelihoodBase,
                 systematics_file: str = None,
                 free_syserr=False):
        
        self.systematics = "fixed"
        if systematics_file is not None:
            self.systematics = "from_file"
            sys_params_per_filter, t_nodes_per_filter, _= process_file(systematics_file, filters=likelihood.filters)
            likelihood._setup_sys_uncertainty_from_file(sys_params_per_filter, t_nodes_per_filter)
            self.systematics_file = systematics_file
        
        if free_syserr:
            self.systematics = "free"
            likelihood._setup_sys_uncertainty_free()
        
        self.likelihood = likelihood

        self.tmin = likelihood.data_tmin
        self.tmax = likelihood.data_tmax
        self.times_det = likelihood.times_det
        self.times_nondet = likelihood.times_nondet

        if hasattr(likelihood, "zero_point_mag"):
            def flux_to_mag(flux_arr):
                return -2.5*np.log10(flux) + likelihood.zero_point_mag
            self.mag_det = jax.tree.map(flux_to_mag, likelihood.datapoints_det)
            self.mag_nondet = jax.tree.map(flux_to_mag, likelihood.datapoints_nondet)
            self.mag_err = -2.5 * jax.tree.map(lambda x, y: x/y, likelihood.datapoints_err, likelihood.datapoints_det)

        else:
            self.mag_det = likelihood.datapoints_det
            self.mag_err = likelihood.datapoints_err
            self.mag_nondet = likelihood.datapoints_nondet
        
        self.model = likelihood.model
        self.posterior = pd.DataFrame(posterior)
        self.fixed_params = likelihood.fixed_params

    def plot_data(self, 
                  ax: matplotlib.axes.Axes, 
                  filt: str,
                  zorder=3,
                  **kwargs):
        """
        Plots data points from a filter over ax.

        Args:
            ax (matplotlib.axes.Axes): ax to plot the data points to.
            filt (str): Which filter from the data should be plotted on ax.
            zorder (int): zorder with which the data points should be plotted.
            **kwargs: kwargs to be passed to errorbar and scatter.
        """
            
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
        """
        Plots one filter from the best fit light curve from the posterior over ax.

        Args:
            ax (matplotlib.axes.Axes): ax to plot the light curve onto.
            filt (str): Which filter from the best fit lightcurve should be plotted on ax.
            zorder (int): zorder with which the lightcurve should be plotted.
            **kwargs: kwargs to be passed to plot.
        """

        self._get_best_fit_lc()
        ax.plot(self.t_best_fit, self.best_fit_lc[filt], zorder=zorder, **kwargs)
        
    
    def _get_best_fit_lc(self,):

        if hasattr(self, "_best_fit_lc_determined"):
            return

        best_ind = np.argmax(self.posterior["log_likelihood"])
        self.best_fit_params = self.posterior.iloc[best_ind].to_dict()
        
        self.best_fit_params.update(self.fixed_params)
        self.best_fit_params = self.likelihood.conversion(self.best_fit_params)

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
        """
        Plots background light curves from the posterior over ax.

        Args:
            ax (matplotlib.axes.Axes): ax to plot the light curve onto.
            filt (str): Which filter from the background light curves should be plotted on ax.
            zorder (int): zorder with which the lightcurve should be plotted.
        """
        
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
            params[key] = self.posterior[key][ind].to_numpy()
        for key in self.fixed_params:
            params[key] = np.ones(200) * self.fixed_params[key]
        
        params = self.likelihood.conversion(params)
        self.t_sample_lc, self.sample_lc = self.model.vpredict(params)
        self._sample_lcs_determined = True

    def plot_sys_uncertainty_band(self,
                                  ax: matplotlib.axes.Axes,
                                  filt: str,
                                  zorder=2,
                                  **kwargs):
        """
        Plots systematic uncertainty band from the best fit light curve for one filter over ax.

        Args:
            ax (matplotlib.axes.Axes): ax to plot the band onto.
            filt (str): Which filter from the band should be plotted on ax.
            zorder (int): zorder with which the band should be plotted.
            **kwargs: kwargs to be passed to fill_between.
        """
        self._get_best_fit_lc()

        if self.systematics=="from_file":

            sys_params_per_filter, t_nodes_per_filter, _ = process_file(self.systematics_file, [filt])
            sys_params_per_filter = sys_params_per_filter[filt]
            t_nodes_per_filter = t_nodes_per_filter[filt]
    
            if t_nodes_per_filter is None:
                t_nodes_per_filter = np.linspace(self.tmin, self.tmax, len(sys_params_per_filter))
            
            
            sys_param_array = np.array([self.best_fit_params[p] for p in sys_params_per_filter])
            sigma_sys = np.interp(self.t_best_fit, t_nodes_per_filter, sys_param_array)

        elif self.systematics=="free":

            sigma_sys = self.best_fit_params["sys_err"]
        
        else:
            sigma_sys  = self.likelihood.error_budget

        ax.fill_between(self.t_best_fit, self.best_fit_lc[filt] + sigma_sys, self.best_fit_lc[filt] - sigma_sys, alpha=0.1, zorder=zorder, **kwargs)
    
    def get_chisquared(self, per_dof: bool=False):
        """
        Get the total chisquared value and the chisquared values per filter. 
        This is different from the log_likelihood value in the posterior, because the likelihood function contains (2 pi sigma)^(-1/2).

        Args:
            per_dof (bool): Whether to return reduced chi-squared values, i.e., per number of data points.
        
        Returns:
            tuple(float, dict): The total chi-squared value across all data points and a dict with the chi-squared value in each filter.
        """
       
        self._get_best_fit_lc()
        
        mag_est_det = jax.tree.map(lambda t, m: jnp.interp(t, self.t_best_fit, m, left = "extrapolate", right = "extrapolate"), # TODO extrapolation is maybe problematic here
                                          self.times_det, self.best_fit_lc)
        
        mag_est_nondet = jax.tree.map(lambda t, m: jnp.interp(t, self.t_best_fit, m, left = "extrapolate", right = "extrapolate"),
                                          self.times_nondet, self.best_fit_lc)
        
        # Get the systematic uncertainty + data uncertainty
        sigma = self.likelihood.get_sigma(self.best_fit_params)
        nondet_sigma = self.likelihood.get_nondet_sigma(self.best_fit_params)
        
        # Get chisq
        chisq_dict = jax.tree.map(lambda mstar, md, s: jnp.sum((mstar-md)**2/s**2), 
                             mag_est_det, self.mag_det, sigma)
        
        chisq_total = sum(chisq_dict.values())
        

        if per_dof:
            n_data_total = 0
            for key in self.mag_det.keys():
                n_data = self.mag_det[key].shape[0]
                chisq_dict[key] = chisq_dict[key] / n_data
                n_data_total += n_data

            chisq_total /= n_data_total
        
        chisq_dict["chisqu_total"] = chisq_total

        return chisq_dict

        
                                  
