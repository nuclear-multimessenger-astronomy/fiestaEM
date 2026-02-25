import os
import ast
import warnings

import h5py
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable

from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

from fiesta.inference.lightcurve_model import LightcurveModel, FluxModel
from fiesta.plot import latex_labels

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
            self.metric_name = "$\\mathcal{L}_\\infty$"
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
            self.parameter_distributions = self.model.parameter_distributions
            self.parameter_names =  self.model.parameter_names
            nus = f["nus"][:]

            self.test_X_raw = f["test"]["X"][:]
            test_y_raw = f["test"]["y"][:]
            test_y_raw = test_y_raw.reshape(len(self.test_X_raw), len(f["nus"]), len(f["times"]) )

            test_y_raw = interp1d(f["times"][:], test_y_raw, axis = 2)(self.times) # interpolate the test data over the time range of the model
            self.test_log_flux = test_y_raw  # store log10 flux for FluxModel error calculation
            self.data_nus = nus  # store raw data frequency grid
            mJys = np.power(10, test_y_raw)
        
        if "redshift" in self.parameter_names:
            from fiesta.train.DataManager import concatenate_redshift, redshifted_magnitude
            self.test_X_raw = concatenate_redshift(self.test_X_raw, max_z=self.parameter_distributions["redshift"][1])
            for Filt in self.Filters:
                self.test_mag[Filt.name] = jnp.array(redshifted_magnitude(Filt, mJys.copy(), nus, self.test_X_raw[:,-1]))
        else:
            for Filt in self.Filters:
                self.test_mag[Filt.name] = Filt.get_mags(mJys, nus)
        
        # get the model prediction on the test data
        param_dict = dict(zip(self.parameter_names, self.test_X_raw.T))
        param_dict["luminosity_distance"] = np.ones(len(self.test_X_raw)) * 1e-5
        if "redshift" not in param_dict.keys():
            param_dict["redshift"] = np.zeros(len(self.test_X_raw))
        _, self.pred_mag = self.model.vpredict(param_dict)         
    
    def calculate_error(self,):
        self.error = {}

        for Filt in self.Filters:
            test_y = self.test_mag[Filt.name]
            pred_y = self.pred_mag[Filt.name]
            mask = np.isinf(pred_y) | np.isinf(test_y)
            test_y = test_y.at[mask].set(0.)
            pred_y = pred_y.at[mask].set(0.)
            self.error[Filt.name] = self.metric(test_y - pred_y)

        if isinstance(self.model, FluxModel):
            self.nus = self.model.nus
            log_flux_pred = []
            for j in range(len(self.test_X_raw)):
                param_dict_j = dict(zip(self.parameter_names, self.test_X_raw[j], strict=True))
                param_dict_j["luminosity_distance"] = 1e-5
                param_dict_j["redshift"] = 0.0
                _, pred_nus, log_flux = self.model.predict_log_flux(param_dict_j)
                log_flux_pred.append(log_flux)
            log_flux_pred = np.array(log_flux_pred)
            # Interpolate ground truth onto the prediction's nu/time grid
            pred_nus = np.array(pred_nus)
            test_log_interp = interp1d(self.data_nus, self.test_log_flux, axis=1,
                                       bounds_error=False, fill_value=np.nan)(pred_nus)
            log_flux_residual = log_flux_pred - test_log_interp
            # Mask non-finite entries before clipping
            nan_mask = ~np.isfinite(log_flux_residual)
            n_nan = np.count_nonzero(nan_mask)
            n_total = log_flux_residual.size
            self.nan_fraction = n_nan / n_total if n_total > 0 else 0.0
            if n_nan > 0:
                warnings.warn(
                    f"Benchmarker: {n_nan}/{n_total} ({100*self.nan_fraction:.1f}%) "
                    f"residual entries are NaN/Inf (likely from frequency grid "
                    f"extrapolation). These entries are excluded from the total "
                    f"error calculation.",
                    stacklevel=2)
            # Set non-finite entries to NaN, then clip physical residuals
            log_flux_residual = np.where(nan_mask, np.nan, log_flux_residual)
            log_flux_residual = np.clip(log_flux_residual, -100, 100)
            # Exclude NaN/Inf entries from error calculation
            if self.file_ending == "Linf":
                self.error["total"] = np.nanmax(np.abs(log_flux_residual), axis=(1, 2))
            else:
                # NaN-aware L2: integrate only over finite entries per sample
                r2 = np.where(nan_mask, np.nan, log_flux_residual ** 2)
                self.error["total"] = np.sqrt(np.nanmean(r2, axis=(1, 2)))
            # Replace NaN/Inf (from all-NaN samples or overflow) with 0
            self.error["total"] = np.nan_to_num(
                self.error["total"], nan=0.0, posinf=0.0, neginf=0.0)
        else:
            max_errors = {key: np.max(value) for key, value in self.error.items()}
            max_key = max(max_errors, key=max_errors.get)
            self.error["total"] = self.error[max_key]
    
    def get_error_distribution(self,):
        error_distribution = {}
        # Normalize weights to prevent overflow in density computation
        total = self.error["total"]
        w_max = np.max(np.abs(total))
        if w_max > 0:
            weights = total / w_max
        else:
            weights = np.ones_like(total)
        for j, p in enumerate(self.parameter_names):
            p_array = self.test_X_raw[:,j]
            bins = np.linspace(self.parameter_distributions[p][0], self.parameter_distributions[p][1], 12)
            error_distribution[p] = np.histogram(p_array, weights=weights, bins=bins, density=True)

        self.error_distribution = error_distribution
    
    def benchmark(self,):
        self.print_correlations()
        self.plot_worst_lightcurves()
        self.plot_error_over_time()
        self.plot_error_distribution()

    def plot_lightcurves_mismatch(self):
        if self.metric_name == "$\\mathcal{L}_2$":
            vline = self.metric(np.ones(len(self.times)))
            vmin, vmax = 0, vline*2
            bins = np.linspace(vmin, vmax, 25)
        else:
            vline = 1.
            vmin, vmax = 0, 2*vline
            bins = np.linspace(vmin, vmax, 20)
    
        cmap = colors.LinearSegmentedColormap.from_list(name = "mymap", colors = [(0, "lightblue"), (1, "darkred")])
        label_dic = {p: latex_labels.get(p, p) for p in self.parameter_names}

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
            plt.close(fig)
    
    def plot_worst_lightcurves(self,):
        label_dic = {p: latex_labels.get(p, p) for p in self.parameter_names}

        MAG_FAINT_CLIP = 40  # magnitudes fainter than this are unphysical

        n_filters = len(self.Filters)
        ncols = min(n_filters, 3)
        nrows = int(np.ceil(n_filters / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
        axes = np.atleast_2d(axes)
        fig.subplots_adjust(hspace=0.55, wspace=0.35, bottom=0.06, top=0.94, left=0.07, right=0.97)

        for i, filt in enumerate(self.Filters):
            cax = axes[i // ncols, i % ncols]
            ind = np.argmax(self.error[filt.name])
            prediction = np.array(self.pred_mag[filt.name][ind])
            truth = np.array(self.test_mag[filt.name][ind])

            cax.plot(self.times, truth, color="red", lw=1.8, label="Baseline", zorder=3)
            cax.plot(self.times, prediction, color="royalblue", lw=1.0, alpha=0.85, label="Surrogate", zorder=2)
            cax.fill_between(self.times, prediction - 1, prediction + 1,
                             color="royalblue", alpha=0.12, zorder=1)

            # Y-limits from truth only, clamped to physical range
            truth_finite = truth[np.isfinite(truth)]
            truth_clipped = truth_finite[truth_finite < MAG_FAINT_CLIP]
            if len(truth_clipped) > 0:
                ylo = np.min(truth_clipped)
                yhi = np.max(truth_clipped)
            elif len(truth_finite) > 0:
                ylo, yhi = np.min(truth_finite), MAG_FAINT_CLIP
            else:
                ylo, yhi = -5, MAG_FAINT_CLIP
            pad = max(2.0, (yhi - ylo) * 0.12)
            cax.set_ylim(yhi + pad, ylo - pad)  # inverted for magnitudes

            cax.set(xscale="log", xlim=(self.times[0], self.times[-1]))
            cax.set_xlabel("$t$ [days]", fontsize=9)
            cax.set_ylabel("mag", fontsize=9)
            cax.set_title(filt.name, fontsize=11, fontweight="bold")
            cax.grid(True, alpha=0.25, lw=0.5)
            cax.tick_params(labelsize=8)

            # Multi-line parameter annotation (4 params per line)
            params_per_line = 4
            items = [f"{label_dic.get(p, p)}={self.test_X_raw[ind, j]:.2g}"
                     for j, p in enumerate(self.parameter_names)]
            lines = [", ".join(items[k:k + params_per_line])
                     for k in range(0, len(items), params_per_line)]
            param_str = "\n".join(lines)
            cax.text(0.03, 0.04, param_str, transform=cax.transAxes,
                     fontsize=6.5, color="0.35", va="bottom", family="monospace",
                     bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8",
                               pad=2, boxstyle="round,pad=0.3"))

            if i == 0:
                cax.legend(fontsize=9, loc="upper right",
                           framealpha=0.9, edgecolor="0.8")

        for i in range(n_filters, nrows * ncols):
            axes[i // ncols, i % ncols].set_visible(False)

        fig.savefig(os.path.join(self.outdir, f"worst_lightcurves_{self.file_ending}.pdf"), dpi=200)
        plt.close(fig)

    def plot_error_over_time(self,):
        n_filters = len(self.Filters)
        ncols = min(n_filters, 3)
        nrows = int(np.ceil(n_filters / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
        axes = np.atleast_2d(axes)
        fig.subplots_adjust(hspace=0.55, wspace=0.35, bottom=0.06, top=0.94, left=0.07, right=0.97)

        # Pick time indices evenly in log-space
        log_times = np.log10(self.times)
        target_log = np.linspace(log_times[0], log_times[-1], 10)
        indices = np.array([np.argmin(np.abs(log_times - t)) for t in target_log])
        indices = np.unique(indices)

        for i, filt in enumerate(self.Filters):
            cax = axes[i // ncols, i % ncols]
            error = np.abs(np.array(self.pred_mag[filt.name]) - np.array(self.test_mag[filt.name]))
            error = np.where(np.isfinite(error), error, 0.0)

            # Clip outliers at 99th percentile across all times for cleaner violins
            all_err = error[:, indices].ravel()
            clip_val = np.percentile(all_err[all_err > 0], 99) if np.any(all_err > 0) else 1.0
            error_clipped = np.clip(error, 0, clip_val)

            # Use log-space positions for the violin plot
            log_pos = np.log10(self.times[indices])
            spacing = np.diff(np.concatenate([[log_pos[0] - 0.5], log_pos]))
            width = spacing * 0.55
            width = np.clip(width, 0.08, None)

            data_list = [error_clipped[:, idx] for idx in indices]
            parts = cax.violinplot(data_list, positions=log_pos, widths=width, points=300,
                                   showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor("steelblue")
                pc.set_edgecolor("steelblue")
                pc.set_alpha(0.5)
            parts["cmedians"].set_color("darkred")
            parts["cmedians"].set_linewidth(1.5)

            # Manual log-scale tick labels
            tick_vals = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])
            tick_vals = tick_vals[(tick_vals >= self.times[0]) & (tick_vals <= self.times[-1])]
            cax.set_xticks(np.log10(tick_vals))
            cax.set_xticklabels([f"$10^{{{int(np.log10(v))}}}$" for v in tick_vals],
                                fontsize=8)
            cax.set_xlim(log_times[0] - 0.3, log_times[-1] + 0.3)

            # Y-limit from the clipped data median + a few sigma
            medians = np.array([np.median(d) for d in data_list])
            p90 = np.percentile(error_clipped[:, indices].ravel(), 90)
            cax.set_ylim(0, max(p90 * 1.5, np.max(medians) * 3, 0.5))

            cax.set_xlabel("$t$ [days]", fontsize=9)
            cax.set_ylabel("error [mag]", fontsize=9)
            cax.set_title(filt.name, fontsize=11, fontweight="bold")
            cax.grid(True, axis="y", alpha=0.25, lw=0.5)
            cax.tick_params(labelsize=8)

        for i in range(n_filters, nrows * ncols):
            axes[i // ncols, i % ncols].set_visible(False)

        fig.savefig(os.path.join(self.outdir, "error_over_time.pdf"), dpi=200)
        plt.close(fig)

    def print_correlations(self, ):
        for Filt in self.Filters:
            error = self.error[Filt.name]
            print(f"\n \n \nCorrelations for filter {Filt.name}:\n")
            for j, p in enumerate(self.parameter_names):
                print(f"{p}: {np.corrcoef(self.test_X_raw[:,j], error)[0,1]}")
    
    def plot_error_distribution(self,):
        label_dic = {p: latex_labels.get(p, p) for p in self.parameter_names}

        n_params = len(self.parameter_names)
        ncols = min(n_params, 4)
        nrows = int(np.ceil(n_params / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
        axes = np.atleast_2d(axes)
        fig.subplots_adjust(hspace=0.6, wspace=0.4, bottom=0.10, top=0.90, left=0.07, right=0.97)

        nan_frac = getattr(self, 'nan_fraction', 0.0)
        title = "Total error distribution per parameter"
        if nan_frac > 0:
            title += f"  ({100*nan_frac:.1f}% of flux residual entries were NaN/Inf, excluded)"
        fig.suptitle(title, fontsize=10)

        for j, p in enumerate(self.parameter_names):
            cax = axes[j // ncols, j % ncols]
            p_array = self.test_X_raw[:, j]
            pmin, pmax = self.parameter_distributions[p][0], self.parameter_distributions[p][1]
            bins = np.linspace(pmin, pmax, 15)

            # Mean error per bin
            counts, _ = np.histogram(p_array, bins=bins)
            weighted, _ = np.histogram(p_array, bins=bins, weights=self.error["total"])
            mean_error = np.where(counts > 0, weighted / counts, 0)

            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            cax.bar(bin_centers, mean_error, width=np.diff(bins) * 0.85,
                    color="steelblue", edgecolor="white", linewidth=0.5)
            cax.set_xlabel(label_dic.get(p, p), fontsize=9)
            cax.set_ylabel(f"mean {self.metric_name}", fontsize=9)
            cax.set_xlim(pmin, pmax)
            cax.grid(True, axis="y", alpha=0.25, lw=0.5)
            cax.tick_params(labelsize=8)

        for i in range(n_params, nrows * ncols):
            axes[i // ncols, i % ncols].set_visible(False)

        fig.savefig(os.path.join(self.outdir, "error_distribution.pdf"), dpi=200)
        plt.close(fig)
