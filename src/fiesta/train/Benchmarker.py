import os
import warnings

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

from fiesta.logging import logger
from fiesta.train.DataLoader import DataLoader, concatenate_redshift, redshifted_magnitude


def _mean_square_lc_error(times, residual):
    return np.sqrt(trapezoid(x=np.log(times), y=residual**2, axis=-1)) / (np.log(times[-1]) - np.log(times[0]))


def _highest_lc_error(times, residual):
    return np.max(np.abs(residual), axis=-1)


# The two error metrics that ``Benchmarker.benchmark()`` evaluates and plots for every filter.
METRICS = {
    "mean_square_lc_error": {"latex": "$\\mathcal{L}_2$", "func": _mean_square_lc_error},
    "highest_lc_error": {"latex": "$\\mathcal{L}_\\infty$", "func": _highest_lc_error},
}


class Benchmarker:

    def __init__(self,
                 model,
                 data: DataLoader,
                 filters: list = None,
                 outdir: str = "./benchmarks",
                 output_format: str = "pdf",
                 ) -> None:

        self.model = model
        self.times = self.model.times
        self.data = data
        self.outdir = outdir
        self.output_format = output_format

        # Load filters
        if filters is None:
            self.Filters = model.Filters
        else:
            self.Filters = [Filt for Filt in model.Filters if Filt.name in filters]

        # load data and compute both error metrics for every filter
        self.get_data()
        self.calculate_error()

        # steal latex_labels from inference
        # this is a dirty fix for a circular import issue
        from fiesta.inference.plot import latex_labels
        self.latex_labels = latex_labels

        logger.info(f"Initialized benchmarker for model {self.model}.")
        logger.info(f"Loaded filters are: {[Filt.name for Filt in self.Filters]}.")

    def get_data(self,):


        self.parameter_names = self.model.parameter_names
        self.parameter_distributions = self.model.parameter_distributions

        test_X_raw, test_y_raw = self.data.load_from_file("test", slice(None, None))
        test_y_raw = test_y_raw.reshape(len(test_X_raw), self.data.n_nus, self.data.n_times)
        test_y_raw = interp1d(self.data.times, test_y_raw, axis=2)(self.times) # interpolate the test data over the time range of the model

        self.test_X_raw = test_X_raw
        self.test_log_flux = test_y_raw  # store log10 flux for FluxModel error calculation
        self.data_nus = self.data.nus  # store data frequency grid
        mJys = np.power(10, test_y_raw)

        self.test_mag = {}
        if "redshift" in self.parameter_names:
            self.test_X_raw = concatenate_redshift(self.test_X_raw, max_z=self.parameter_distributions["redshift"][1])
            for Filt in self.Filters:
                self.test_mag[Filt.name] = jnp.array(redshifted_magnitude(Filt, mJys.copy(), self.data_nus, self.test_X_raw[:,-1]))
        else:
            for Filt in self.Filters:
                self.test_mag[Filt.name] = Filt.get_mags(mJys, self.data_nus)

        # get the model prediction on the test data
        param_dict = dict(zip(self.parameter_names, self.test_X_raw.T))
        param_dict["luminosity_distance"] = np.ones(len(self.test_X_raw)) * 1e-5
        if "redshift" not in param_dict.keys():
            param_dict["redshift"] = np.zeros(len(self.test_X_raw))
        _, self.pred_mag = self.model.vpredict(param_dict)

    def calculate_error(self,):
        self.error = {metric_key: {} for metric_key in METRICS}

        for Filt in self.Filters:
            test_y = self.test_mag[Filt.name]
            pred_y = self.pred_mag[Filt.name]
            mask = np.isinf(pred_y) | np.isinf(test_y)
            test_y = test_y.at[mask].set(0.)
            pred_y = pred_y.at[mask].set(0.)
            residual = test_y - pred_y
            for metric_key, metric in METRICS.items():
                self.error[metric_key][Filt.name] = metric["func"](self.times, residual)

        if hasattr(self.model, "nus"):
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
            r2 = np.where(nan_mask, np.nan, log_flux_residual ** 2)
            totals = {
                "highest_lc_error": np.nanmax(np.abs(log_flux_residual), axis=(1, 2)),
                "mean_square_lc_error": np.sqrt(np.nanmean(r2, axis=(1, 2))),
            }
            for metric_key, total in totals.items():
                # Replace NaN/Inf (from all-NaN samples or overflow) with 0
                self.error[metric_key]["total"] = np.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            for metric_key in METRICS:
                max_errors = {key: np.max(value) for key, value in self.error[metric_key].items()}
                max_key = max(max_errors, key=max_errors.get)
                self.error[metric_key]["total"] = self.error[metric_key][max_key]

    ###############################
    # ACTUAL BENCHMARKING METHODS #
    ###############################

    def benchmark(self,):
        self.plot_error_over_time()
        self.plot_worst_lightcurves()
        self.plot_lightcurves_mismatch()

    def plot_worst_lightcurves(self,):

        for metric in METRICS:
            self.worst_lightcurves(metric)       

    def worst_lightcurves(self, metric_key: str = "highest_lc_error"):
        label_dic = {p: self.latex_labels.get(p, p) for p in self.parameter_names}

        MAG_FAINT_CLIP = 40  # magnitudes fainter than this are unphysical

        n_filters = len(self.Filters)
        ncols = min(n_filters, 3)
        nrows = int(np.ceil(n_filters / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
        axes = np.atleast_2d(axes)
        fig.subplots_adjust(hspace=0.55, wspace=0.35, bottom=0.06, top=0.94, left=0.07, right=0.97)

        for i, filt in enumerate(self.Filters):
            cax = axes[i // ncols, i % ncols]
            ind = np.argmax(self.error[metric_key][filt.name])
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

        fig.suptitle(metric_key)

        fig.savefig(os.path.join(self.outdir, f"worst_lightcurves_{metric_key}.{self.output_format}"), dpi=200)
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

        fig.savefig(os.path.join(self.outdir, f"benchmark_error_over_time.{self.output_format}"), dpi=200)
        plt.close(fig)

    def print_correlations(self, metric_key: str = "highest_lc_error"):
        for Filt in self.Filters:
            error = self.error[metric_key][Filt.name]
            print(f"\n \n \nCorrelations for filter {Filt.name}:\n")
            for j, p in enumerate(self.parameter_names):
                print(f"{p}: {np.corrcoef(self.test_X_raw[:,j], error)[0,1]}")

    def plot_lightcurves_mismatch(self,):

        for metric in METRICS:
            self.lightcurves_mismatch(metric)


    def lightcurves_mismatch(self, metric_key: str = "highest_lc_error"):

        if metric_key == "mean_square_lc_error":
            vline = METRICS[metric_key]["func"](self.times, np.ones(len(self.times)))
            vmin, vmax = 0, vline*2
            bins = np.linspace(vmin, vmax, 25)
        else:
            vline = 1.
            vmin, vmax = 0, 2*vline
            bins = np.linspace(vmin, vmax, 20)

        cmap = colors.LinearSegmentedColormap.from_list(name = "mymap", colors = [(0, "lightblue"), (1, "darkred")])
        label_dic = {p: self.latex_labels.get(p, p) for p in self.parameter_names}

        n_params = len(self.parameter_names)
        # size of the pairwise-scatter corner grid; at least 1x1 even for a single parameter
        n_grid = max(n_params - 1, 1)

        for Filt in self.Filters:

            mismatch = self.error[metric_key][Filt.name]
            colored_mismatch = cmap(mismatch/vmax)

            # the histogram always gets its own dedicated axis (an extra column), rather
            # than reusing a "spare" corner-grid cell, since for n_params <= 2 the corner
            # grid has no spare cell to give it
            fig = plt.figure(figsize=(2.6 * (n_grid + 1), 2.6 * n_grid))
            gs = fig.add_gridspec(n_grid, n_grid + 1)
            ax = np.empty((n_grid, n_grid), dtype=object)
            for r in range(n_grid):
                for c in range(n_grid):
                    ax[r, c] = fig.add_subplot(gs[r, c])
            hist_ax = fig.add_subplot(gs[0, -1])

            fig.suptitle(f"{Filt.name}: {METRICS[metric_key]['latex']} norm")

            sort = np.argsort(mismatch)
            if n_params == 1:
                p = self.parameter_names[0]
                ax[0,0].scatter(self.test_X_raw[sort,0], mismatch[sort], c = colored_mismatch[sort], s = 1, rasterized = True)
                ax[0,0].set_xlim((self.test_X_raw[:,0].min(), self.test_X_raw[:,0].max()))
                ax[0,0].set_xlabel(label_dic[p])
                ax[0,0].set_ylabel(METRICS[metric_key]['latex'])
            else:
                for j, p in enumerate(self.parameter_names[1:]):
                    for k, pp in enumerate(self.parameter_names[:j+1]):

                        ax[j,k].scatter(self.test_X_raw[sort,k], self.test_X_raw[sort,j+1], c = colored_mismatch[sort], s = 1, rasterized = True)

                        ax[j,k].set_xlim((self.test_X_raw[:,k].min(), self.test_X_raw[:,k].max()))
                        ax[j,k].set_ylim((self.test_X_raw[:,j+1].min(), self.test_X_raw[:,j+1].max()))


                        if k!=0:
                            ax[j,k].set_yticklabels([])

                        if j!=n_grid-1:
                            ax[j,k].set_xticklabels([])

                        ax[-1,k].set_xlabel(label_dic[pp])
                    ax[j,0].set_ylabel(label_dic[p])

                    for cax in ax[j, j+1:]:
                        cax.set_axis_off()

            hist_ax.hist(mismatch, density = True, histtype = "step", bins = bins,)
            hist_ax.vlines([vline], *hist_ax.get_ylim(), colors = ["lightgrey"], linestyles = "dashed")
            hist_ax.set_yticks([])
            hist_ax.set_xlabel(METRICS[metric_key]['latex'])

            outfile  = f"benchmark_{Filt.name}_{metric_key}.{self.output_format}"

            fig.suptitle(metric_key)

            fig.savefig(os.path.join(self.outdir, outfile))
            plt.close(fig)

    def plot_error_distribution(self, metric_key: str = "highest_lc_error"):
        label_dic = {p: self.latex_labels.get(p, p) for p in self.parameter_names}

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

        total_error = self.error[metric_key]["total"]
        for j, p in enumerate(self.parameter_names):
            cax = axes[j // ncols, j % ncols]
            p_array = self.test_X_raw[:, j]
            pmin, pmax = self.parameter_distributions[p][0], self.parameter_distributions[p][1]
            bins = np.linspace(pmin, pmax, 15)

            # Mean error per bin
            counts, _ = np.histogram(p_array, bins=bins)
            weighted, _ = np.histogram(p_array, bins=bins, weights=total_error)
            mean_error = np.where(counts > 0, weighted / counts, 0)

            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            cax.bar(bin_centers, mean_error, width=np.diff(bins) * 0.85,
                    color="steelblue", edgecolor="white", linewidth=0.5)
            cax.set_xlabel(label_dic.get(p, p), fontsize=9)
            cax.set_ylabel(f"mean {METRICS[metric_key]['latex']}", fontsize=9)
            cax.set_xlim(pmin, pmax)
            cax.grid(True, axis="y", alpha=0.25, lw=0.5)
            cax.tick_params(labelsize=8)

        for i in range(n_params, nrows * ncols):
            axes[i // ncols, i % ncols].set_visible(False)

        fig.savefig(os.path.join(self.outdir, f"error_distribution_{metric_key}.{self.output_format}"), dpi=200)
        plt.close(fig)
