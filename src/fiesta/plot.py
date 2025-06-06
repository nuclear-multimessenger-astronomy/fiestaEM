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
        "figure.titlesize": 16}
plt.rcParams.update(pltparams)

import numpy as np

from jaxtyping import Array
from fiesta.logging import logger

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
                  v_ej_dyn="$\\bar{v}_{\\mathrm{ej,dyn}})$",
                  v_ej_wind="$\\bar{v}_{\\mathrm{ej,wind}})$",
                  Ye_dyn="$\\bar{Y}_{e,\\mathrm{dyn}}$",
                  Ye_wind="$Y_{e,\\mathrm{wind}}$",
                  luminosity_distance="$d_L$",
                  redshift="$z$",
                  sys_err="$\\sigma_{\mathrm{sys}}$")



#######################
#  PLOTTING FUNCTIONS #
#######################


def corner_plot(samples: Array,
            parameter_names: list[str],
            truths: list=None,
            color:str ="blue",
            legend_label:str =None):
    
    try:
        import corner
    except ImportError:
        logger.warning(f"Install corner to create corner plots.")
        return 
    
    labels= []
    for p in parameter_names:
        labels.append(latex_labels.get(p, p))

    if truths is None:
        truths = [None]*samples.shape[1]

    
    fig, ax = plt.subplots(samples.shape[1], samples.shape[1], figsize = (samples.shape[1]*1.5, samples.shape[1]*1.5))
    corner.corner(samples, 
          fig=fig,
          color=color,
          labels=labels,
          truths=truths,
          **default_corner_kwargs,
          hist_kwargs=dict(density=True, color=color))
    
    if legend_label is not None:
        
        if samples.shape[1] < 4:
            lx, ly = 0, -1
        else:
            lx, ly = 1, 4

        handle = plt.plot([],[], color=color)[0]
        ax[lx, ly].legend(handles=[handle], labels=[legend_label], fontsize=15, fancybox=False, framealpha=1)
    
    fig.tight_layout()
    return fig, ax

# TODO: superpose multiple posteriors in one corner plot



