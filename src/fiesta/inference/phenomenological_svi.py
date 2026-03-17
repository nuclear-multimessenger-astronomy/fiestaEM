"""Villar-model SVI fitter with superphot+ priors (de Soto et al. 2024).

Provides ``fit_villar_svi()`` — a self-contained numpyro SVI fitter for
ZTF-style two-band photometry, directly integrated into the fiesta framework.

The core numerics (flux-space likelihood, 14D relative-band parameterization,
TruncatedNormal priors, Villar constraint penalty) are identical to the
standalone fitter in ``fit_jax_best.py`` to ensure reproducible results.

Usage
-----
    from fiesta.inference.villar_svi import fit_villar_svi

    # data: dict mapping filter name to (n, 3) array of [mjd, mag, mag_err]
    result = fit_villar_svi(data, trigger_time=t0)
    print(result["best_fit"])
    print(result["samples"])
"""

import logging
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import lax, random
from numpyro.infer import SVI, Trace_ELBO

logger = logging.getLogger(__name__)

# ── Filter aliases ───────────────────────────────────────────────────────────
FILTER_ALIASES = {
    "ZTF_r": "ZTF_r", "ZTF_g": "ZTF_g",
    "ztfr": "ZTF_r", "ztfg": "ZTF_g",
    "r": "ZTF_r", "g": "ZTF_g",
}

# ── Embedded prior tables (de Soto et al. 2024, Table 2) ─────────────────────
# Each entry: (min, max, mean, stddev, logged)
# logged=True → parameter lives in log10 space; 10^x is applied after sampling.

# Villar priors (7 params per band)
_VILLAR_PRIOR_R = {
    "A":           (-0.30,   0.50,  0.0957,  0.150, True),
    "beta":        (-0.01,   0.03,  0.00833, 0.012, False),
    "gamma":       ( 0.00,   3.50,  1.4258,  0.900, True),
    "t_0":         (-100.0,  30.0, -17.878, 30.000, False),
    "tau_rise":    (-2.00,   4.00,  0.6664,  1.200, True),
    "tau_fall":    ( 0.00,   4.00,  1.5261,  0.900, True),
    "extra_sigma": (-3.00,  -0.80, -1.6629,  0.900, True),
}

_VILLAR_PRIOR_G = {
    "A":           (-1.00,   1.00, -0.0766,  0.300, True),
    "beta":        (-0.01,   0.03,  0.0000,  0.010, False),
    "gamma":       (-1.50,   1.50, -0.0452,  0.450, True),
    "t_0":         (-5.00,   5.00, -0.500,   2.500, False),
    "tau_rise":    (-1.50,   1.50, -0.1510,  0.600, True),
    "tau_fall":    (-1.50,   1.50, -0.1486,  0.750, True),
    "extra_sigma": (-1.50,   1.00, -0.1509,  0.750, True),
}

# Bazin priors (5 params per band — Villar without beta, gamma)
_BAZIN_PRIOR_R = {
    "A":           (-0.30,   0.50,  0.0957,  0.150, True),
    "B":           (-0.30,   0.30,  0.0000,  0.100, False),
    "t_0":         (-100.0,  30.0, -17.878, 30.000, False),
    "tau_rise":    (-2.00,   4.00,  0.6664,  1.200, True),
    "tau_fall":    ( 0.00,   4.00,  1.5261,  0.900, True),
    "extra_sigma": (-3.00,  -0.80, -1.6629,  0.900, True),
}

_BAZIN_PRIOR_G = {
    "A":           (-1.00,   1.00, -0.0766,  0.300, True),
    "B":           (-0.30,   0.30,  0.0000,  0.100, False),
    "t_0":         (-5.00,   5.00, -0.500,   2.500, False),
    "tau_rise":    (-1.50,   1.50, -0.1510,  0.600, True),
    "tau_fall":    (-1.50,   1.50, -0.1486,  0.750, True),
    "extra_sigma": (-1.50,   1.00, -0.1509,  0.750, True),
}

# Default to Villar for backwards compat
_PRIOR_R = _VILLAR_PRIOR_R
_PRIOR_G = _VILLAR_PRIOR_G

_BASE_NAMES = list(_PRIOR_R.keys())
FILTERS = ["ZTF_r", "ZTF_g"]
BAND_IDX = {"ZTF_r": 0, "ZTF_g": 1}
N_BASE = len(_BASE_NAMES)
N_PARAMS = 2 * N_BASE


def _build_prior_arrays():
    """Flatten the per-band prior dicts into parallel numpy arrays."""
    mins, maxs, means, stds, is_logged = [], [], [], [], []
    for prior_dict in (_PRIOR_R, _PRIOR_G):
        for name in _BASE_NAMES:
            lo, hi, mean, std, logged = prior_dict[name]
            mins.append(lo)
            maxs.append(hi)
            means.append(mean)
            stds.append(std)
            is_logged.append(logged)
    return (np.array(mins), np.array(maxs), np.array(means),
            np.array(stds), np.array(is_logged, dtype=bool))


MINS, MAXS, MEANS, STDS, LOGGED = _build_prior_arrays()
LOGGED_IDX = np.where(LOGGED)[0]


# ── Photometry helpers ────────────────────────────────────────────────────────

def _mag_to_flux(mag, mag_err, zp=23.90):
    flux = 10.0 ** ((zp - mag) / 2.5)
    flux_err = flux * mag_err * np.log(10.0) / 2.5
    return flux, flux_err


def _merge_close_times(df, eps=0.04):
    parts = []
    for filt, grp in df.groupby("filter"):
        grp = grp.sort_values("phase").reset_index(drop=True)
        i = 0
        rows = []
        while i < len(grp):
            j = i + 1
            while j < len(grp) and grp.at[j, "phase"] - grp.at[i, "phase"] < eps:
                j += 1
            chunk = grp.iloc[i:j]
            w = 1.0 / chunk["flux_err"].values ** 2
            wsum = w.sum()
            rows.append({
                "phase":    (chunk["phase"].values * w).sum() / wsum,
                "flux":     (chunk["flux"].values * w).sum() / wsum,
                "flux_err": np.sqrt(1.0 / wsum),
                "filter":   filt,
            })
            i = j
        parts.append(pd.DataFrame(rows))
    return pd.concat(parts, ignore_index=True)


def _preprocess(df_raw):
    df = df_raw.copy()
    # Normalise filter names
    df["filter"] = df["filter"].map(lambda f: FILTER_ALIASES.get(f, f))
    df = df[df["filter"].isin(FILTERS)].copy()
    if df.empty:
        raise ValueError("No r/g band data.")

    f, fe = _mag_to_flux(df["mag"].values, df["mag_err"].values)
    df["flux"] = f
    df["flux_err"] = fe

    r_rows = df[df["filter"] == "ZTF_r"]
    if r_rows.empty:
        raise ValueError("No r-band data.")
    t0 = r_rows.loc[r_rows["flux"].idxmax(), "mjd"]
    df["phase"] = df["mjd"] - t0

    df = _merge_close_times(df[["phase", "flux", "flux_err", "filter"]])
    df = df[(df["phase"] >= -50.0) & (df["phase"] <= 100.0)].copy()
    df.reset_index(drop=True, inplace=True)

    for filt in FILTERS:
        n = (df["filter"] == filt).sum()
        if n <= 2:
            raise ValueError(f"Only {n} points in {filt} after preprocessing.")

    peak = df["flux"].max()
    if peak <= 0:
        raise ValueError("Non-positive peak flux.")
    df["flux"] /= peak
    df["flux_err"] /= peak

    df_proc = df.copy()
    orig_size = len(df)

    n_total = int(2 ** np.ceil(np.log2(orig_size)))
    n_per_filt = n_total // 2
    pad_rows = []
    for filt in FILTERS:
        n_filt = (df["filter"] == filt).sum()
        for _ in range(max(0, n_per_filt - n_filt)):
            pad_rows.append({"phase": 1000.0, "flux": 0.1, "flux_err": 1000.0, "filter": filt})
    if pad_rows:
        df = pd.concat([df, pd.DataFrame(pad_rows)], ignore_index=True)

    df = df.sort_values(["filter", "phase"]).reset_index(drop=True)

    t = df["phase"].values.astype(np.float32)
    flux = df["flux"].values.astype(np.float32)
    err = df["flux_err"].values.astype(np.float32)
    band = df["filter"].map(BAND_IDX).values.astype(np.int32)

    return t, flux, err, band, orig_size, df_proc


# ── Numpyro model & guide (identical to fit_jax_best.py) ─────────────────────

def _build_param_map(band):
    n = len(band)
    pm = np.zeros((N_BASE, n), dtype=np.int32)
    for i in range(N_BASE):
        for b_idx in range(len(FILTERS)):
            mask = band == b_idx
            pm[i, mask] = i + b_idx * N_BASE
    return jnp.array(pm)


def _villar_flux(flat_params, t, param_map):
    cube = flat_params[param_map]
    amp, beta, gamma, t_0, tau_rise, tau_fall, extra_sigma = cube
    gamma = jnp.clip(gamma, a_min=0.0)
    phase = jnp.clip(t - t_0, a_min=-50.0 * tau_rise)
    f_const = amp / (1.0 + jnp.exp(-phase / tau_rise))
    flux = f_const * jnp.where(
        gamma - phase >= 0,
        1.0 - beta * phase,
        (1.0 - beta * gamma) * jnp.exp(-(phase - gamma) / tau_fall),
    )
    return flux, extra_sigma


def _bazin_flux(flat_params, t, param_map):
    cube = flat_params[param_map]
    amp, baseline, t_0, tau_rise, tau_fall, extra_sigma = cube
    phase = jnp.clip(t - t_0, a_min=-50.0 * tau_rise)
    flux = amp * jnp.exp(-phase / tau_fall) / (1.0 + jnp.exp(-phase / tau_rise)) + baseline
    return flux, extra_sigma


def _villar_constraint(cube):
    return (
        jnp.maximum(cube[2] * cube[1] - 1.0, 0.0)
        + jnp.maximum(jnp.exp(-cube[2] / cube[4]) * (cube[5] / cube[4] - 1.0) - 1.0, 0.0)
        + jnp.maximum(cube[1] * cube[5] - 1.0 + cube[1] * cube[2], 0.0)
    )


def _bazin_constraint(cube):
    # No physical constraints beyond positivity (enforced by priors)
    return 0.0


def _make_numpyro_model(flux_fn, constraint_fn, mins, maxs, means, stds, logged_idx, n_base):
    """Build a numpyro model closure for the given flux function and priors."""
    def model(t, obsflux, uncertainties, param_map):
        min_b = jnp.array(mins[:n_base]);  max_b = jnp.array(maxs[:n_base])
        mu_b  = jnp.array(means[:n_base]); sg_b  = jnp.array(stds[:n_base])
        mu_r  = jnp.array(means[n_base:]); sg_r  = jnp.array(stds[n_base:])

        with numpyro.plate("base_params", n_base):
            base = numpyro.sample(
                "base_samples",
                dist.TruncatedNormal(mu_b, sg_b, low=min_b, high=max_b),
            )

        with numpyro.plate("relative_params", n_base):
            rel = numpyro.sample(
                "relative_samples",
                dist.Normal(base + mu_r, sg_r),
            )

        raw = jnp.concatenate([base, rel])
        lidx = jnp.array(logged_idx)
        flat = raw.at[lidx].set(10.0 ** raw[lidx])

        flux, extra_sigma = flux_fn(flat, t, param_map)
        numpyro.factor("constraint", -10_000.0 * jnp.max(constraint_fn(flat[param_map])))
        sigma = jnp.sqrt(uncertainties ** 2 + extra_sigma ** 2)
        numpyro.sample("obs", dist.Normal(flux, sigma), obs=obsflux)
    return model


def _make_numpyro_guide(mins, maxs, means, stds, n_base):
    """Build a numpyro guide closure for the given priors."""
    def guide(t=None, obsflux=None, uncertainties=None, param_map=None):
        min_b = jnp.array(mins[:n_base]);  max_b = jnp.array(maxs[:n_base])
        mu_b  = jnp.array(means[:n_base]); sg_b  = jnp.array(stds[:n_base])
        min_r = jnp.array(mins[n_base:]);  max_r = jnp.array(maxs[n_base:])
        mu_r  = jnp.array(means[n_base:]); sg_r  = jnp.array(stds[n_base:])

        with numpyro.plate("base_params", n_base):
            loc_b = numpyro.param("loc_base", mu_b,
                                  constraint=dist.constraints.interval(min_b, max_b))
            scale_b = numpyro.param("scale_base", sg_b / 5.0,
                                    constraint=dist.constraints.interval(
                                        1e-5 * jnp.ones(n_base), 3.0 * sg_b))
            numpyro.sample("base_samples", dist.Normal(loc_b, scale_b))

        with numpyro.plate("relative_params", n_base):
            loc_off = numpyro.param("loc_offset", mu_r,
                                    constraint=dist.constraints.interval(min_r, max_r))
            scale_r = numpyro.param("scale_relative", sg_r / 5.0,
                                    constraint=dist.constraints.interval(
                                        1e-6 * jnp.ones(n_base), 3.0 * sg_r))
            numpyro.sample("relative_samples", dist.Normal(loc_b + loc_off, scale_r))
    return guide


# ── Scoring ──────────────────────────────────────────────────────────────────

def _score_villar(cube, t_b):
    amp, beta, gamma, t_0, tau_rise, tau_fall, extra_sigma = cube
    gamma_c = np.clip(gamma, 0.0, None)
    phase = np.clip(t_b - t_0, a_min=-50.0 * tau_rise, a_max=None)
    f_model = amp / (1.0 + np.exp(-phase / tau_rise))
    f_model = np.where(
        gamma_c - phase >= 0,
        f_model * (1.0 - beta * phase),
        f_model * (1.0 - beta * gamma_c) * np.exp(-(phase - gamma_c) / tau_fall),
    )
    return f_model, extra_sigma


def _score_bazin(cube, t_b):
    amp, baseline, t_0, tau_rise, tau_fall, extra_sigma = cube
    phase = np.clip(t_b - t_0, a_min=-50.0 * tau_rise, a_max=None)
    f_model = amp * np.exp(-phase / tau_fall) / (1.0 + np.exp(-phase / tau_rise)) + baseline
    return f_model, extra_sigma


def _score_samples(samples_df, t, flux, err, param_map, orig_size, score_fn, n_base, n_params):
    flat = samples_df.values.astype(np.float32)
    pm = np.array(param_map)
    cube = flat.T[pm]
    t_b = np.array(t, dtype=np.float32)[:, np.newaxis]

    f_model, extra_sigma = score_fn(cube, t_b)

    e_b = np.array(err, dtype=np.float32)[:, np.newaxis]
    sigma2 = e_b ** 2 + extra_sigma ** 2
    f_b = np.array(flux, dtype=np.float32)[:, np.newaxis]
    dof = max(orig_size - n_params, 1)
    return np.sum((f_b - f_model) ** 2 / sigma2, axis=0) / dof


# ── Generic SVI runner ───────────────────────────────────────────────────────

def _prepare_data(data):
    """Convert input data to a preprocessed DataFrame."""
    if isinstance(data, pd.DataFrame):
        df_raw = data.copy()
    else:
        frames = []
        for filt, arr in data.items():
            canon = FILTER_ALIASES.get(filt, filt)
            filt_label = {"ZTF_r": "r", "ZTF_g": "g"}.get(canon, filt)
            df = pd.DataFrame(arr, columns=["mjd", "mag", "mag_err"])
            df["filter"] = filt_label
            frames.append(df)
        df_raw = pd.concat(frames, ignore_index=True)

    filt_map = {"r": "ZTF_r", "g": "ZTF_g", "ztfr": "ZTF_r", "ztfg": "ZTF_g",
                "ZTF_r": "ZTF_r", "ZTF_g": "ZTF_g"}
    df_raw["filter"] = df_raw["filter"].map(lambda f: filt_map.get(f, f))
    return df_raw


def _run_svi(df_raw, prior_r, prior_g, flux_fn, constraint_fn, score_fn,
             model_name, num_iter, step_size, n_samples, seed, outdir,
             score_cutoff=1.2):
    """Core SVI runner shared by fit_villar_svi and fit_bazin_svi."""
    base_names = list(prior_r.keys())
    n_base = len(base_names)
    n_params = 2 * n_base

    # Build prior arrays
    mn, mx, mu, sg, lg = [], [], [], [], []
    for d in (prior_r, prior_g):
        for name in base_names:
            a, b, m, s, l = d[name]
            mn.append(a); mx.append(b); mu.append(m); sg.append(s); lg.append(l)
    mins = np.array(mn); maxs = np.array(mx)
    means = np.array(mu); stds = np.array(sg)
    logged = np.array(lg, dtype=bool)
    logged_idx = np.where(logged)[0]

    t, flux, err, band, orig_size, df_proc = _preprocess(df_raw)

    # Build param map for n_base params
    n_obs = len(band)
    pm = np.zeros((n_base, n_obs), dtype=np.int32)
    for i in range(n_base):
        for b_idx in range(len(FILTERS)):
            mask = band == b_idx
            pm[i, mask] = i + b_idx * n_base
    param_map = jnp.array(pm)

    logger.info("Running %s SVI (%d iterations)…", model_name, num_iter)
    t_j = jnp.array(t); f_j = jnp.array(flux); e_j = jnp.array(err)

    numpyro_model = _make_numpyro_model(flux_fn, constraint_fn, mins, maxs, means, stds, logged_idx, n_base)
    numpyro_guide = _make_numpyro_guide(mins, maxs, means, stds, n_base)

    optimizer = numpyro.optim.Adam(step_size)
    svi = SVI(numpyro_model, numpyro_guide, optimizer, loss=Trace_ELBO())
    rng = random.key(seed)
    svi_state = svi.init(rng, t_j, f_j, e_j, param_map)

    @jax.jit
    def run_loop(init_state):
        def body(state, _):
            new_state, loss = svi.update(state, t_j, f_j, e_j, param_map)
            safe_state = lax.cond(
                jnp.isfinite(loss), lambda: new_state, lambda: state,
            )
            return safe_state, loss
        return lax.scan(body, init_state, None, length=num_iter)

    t0 = time.perf_counter()
    svi_state, losses = run_loop(svi_state)
    elapsed = time.perf_counter() - t0
    logger.info("SVI finished in %.2fs. Final loss: %.2f", elapsed, float(losses[-1]))

    params = svi.get_params(svi_state)
    loc_b = params["loc_base"]
    loc_off = params["loc_offset"]
    loc = jnp.concatenate([loc_b, loc_b + loc_off])
    scale = jnp.concatenate([params["scale_base"], params["scale_relative"]])

    rng2 = random.fold_in(rng, 1)
    draws = np.array(loc) + np.array(
        random.normal(rng2, shape=(n_samples, n_params))
    ) * np.array(scale)
    draws[:, logged_idx] = 10.0 ** draws[:, logged_idx]

    param_names = [f"{p}_ZTF_r" for p in base_names] + [f"{p}_ZTF_g" for p in base_names]
    samples_df = pd.DataFrame(draws, columns=param_names)

    scores = _score_samples(samples_df, t, flux, err, param_map, orig_size, score_fn, n_base, n_params)

    # Filter samples by chi² cutoff (de Soto+ 2024, Section 3.1)
    valid_mask = scores <= score_cutoff if orig_size >= 6 else np.ones(len(scores), dtype=bool)
    if not np.any(valid_mask):
        valid_mask = np.ones(len(scores), dtype=bool)

    best_idx = np.argmin(scores[valid_mask])
    valid_idx = np.where(valid_mask)[0][best_idx]
    best_fit = samples_df.iloc[valid_idx]

    result = {
        "samples": samples_df,
        "best_fit": best_fit,
        "best_chi2": float(scores[valid_idx]),
        "svi_loc": np.array(loc),
        "svi_scale": np.array(scale),
        "scores": scores,
        "df_proc": df_proc,
        "losses": np.array(losses),
        "elapsed_s": elapsed,
        "model": model_name,
    }

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        np.savez(os.path.join(outdir, "posterior.npz"), **{
            col: samples_df[col].values for col in samples_df.columns
        }, scores=scores)

    return result


# ── Public API ───────────────────────────────────────────────────────────────

def fit_villar_svi(data, num_iter=10_000, step_size=0.001,
                   n_samples=1000, seed=42, outdir=None, score_cutoff=1.2):
    """Fit a Villar model to ZTF r+g photometry using numpyro SVI.

    Uses the exact same numerics as ``fit_jax_best.py``: flux-space
    likelihood, 14D relative-band parameterization, superphot+
    TruncatedNormal priors, and Villar constraint penalty.

    Returns dict with 'samples', 'best_fit', 'best_chi2', etc.
    """
    df_raw = _prepare_data(data)
    result = _run_svi(df_raw, _VILLAR_PRIOR_R, _VILLAR_PRIOR_G,
                      _villar_flux, _villar_constraint, _score_villar,
                      "Villar", num_iter, step_size, n_samples, seed, outdir,
                      score_cutoff=score_cutoff)
    return result


def fit_bazin_svi(data, num_iter=10_000, step_size=0.001,
                  n_samples=1000, seed=42, outdir=None, score_cutoff=1.2):
    """Fit a Bazin model to ZTF r+g photometry using numpyro SVI.

    Same approach as ``fit_villar_svi`` but with the Bazin model (6 base
    params: A, B, t_0, tau_rise, tau_fall, extra_sigma) and corresponding
    priors derived from the superphot+ population.

    Returns dict with 'samples', 'best_fit', 'best_chi2', etc.
    """
    df_raw = _prepare_data(data)
    result = _run_svi(df_raw, _BAZIN_PRIOR_R, _BAZIN_PRIOR_G,
                      _bazin_flux, _bazin_constraint, _score_bazin,
                      "Bazin", num_iter, step_size, n_samples, seed, outdir,
                      score_cutoff=score_cutoff)
    return result
