"""Analytical light-curve models implemented in pure JAX.

Each model is fully JIT-compilable and differentiable so that flowMC's MALA
sampler can compute ``jax.grad`` through the likelihood.  The models follow
the same ``predict()`` contract as the surrogate models:

    (source_frame_times, {filter_name: apparent_mag_array})

This makes them drop-in replacements inside ``CombinedSurrogate`` and
``EMLikelihood``.

All internal physics computations use log10 space to avoid float32 overflow
(e.g. explosion energies ~1e49 erg exceed float32 max ~3.4e38).
"""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from fiesta import filters as fiesta_filters
from fiesta.conversions import mag_app_from_mag_abs
from fiesta.constants import (
    c_cgs, msun_cgs, sigSB, kb, Rsun_cgs, days_to_seconds,
    pc_to_cm, h_erg_s, c, au_cgs, km_cgs,
)
from fiesta.logging import logger

# ---------------------------------------------------------------------------
# Pre-computed log10 constants (avoids recomputation inside JIT)
# ---------------------------------------------------------------------------

_LOG10E = jnp.log10(jnp.e)           # log10(e) for converting ln -> log10
_LN10 = jnp.log(10.0)
_LOG10_MSUN = jnp.log10(msun_cgs)    # ~33.30
_LOG10_RSUN = jnp.log10(Rsun_cgs)    # ~10.84
_LOG10_CCGS = jnp.log10(c_cgs)       # ~10.48
_LOG10_SIGMASB = jnp.log10(sigSB)    # ~-4.25
_LOG10_KB = jnp.log10(kb)            # ~-15.86
_LOG10_H_ERG_S = jnp.log10(h_erg_s)  # ~-26.18
_LOG10_4PI = jnp.log10(4.0 * jnp.pi)
_LOG10_PI = jnp.log10(jnp.pi)
_LOG10_D10PC = jnp.log10(10.0 * pc_to_cm)
_LOG10_DAYS2SEC = jnp.log10(days_to_seconds)
_LOG10_AU_CGS = jnp.log10(au_cgs)
_LOG10_KM_CGS = jnp.log10(km_cgs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log10_blackbody_mJy_at_10pc(T, log10_R, nus):
    """log10 of Planck spectral flux density at 10 pc in mJy.

    Works entirely in log10 space for the radius to avoid float32 overflow.

    Parameters
    ----------
    T : scalar  – temperature in K (linear, typically 1e3–1e5, safe for float32)
    log10_R : scalar  – log10 of photospheric radius in cm
    nus : 1-D array  – frequencies in Hz

    Returns
    -------
    log10_mJy : 1-D array (n_nus,) – log10 of spectral flux density in mJy
    """
    # Planck function  B_nu = 2 h nu^3 / c^2  /  (exp(h nu / kT) - 1)
    x = h_erg_s * nus / (kb * T)
    x = jnp.clip(x, 0.0, 80.0)  # prevent overflow in exp

    log10_Bnu = (_LOG10_H_ERG_S + jnp.log10(2.0) + 3.0 * jnp.log10(nus)
                 - 2.0 * _LOG10_CCGS - jnp.log10(jnp.expm1(x)))

    # F_nu = pi * (R / d)^2 * B_nu   (erg/cm^2/s/Hz at 10 pc)
    log10_Fnu = _LOG10_PI + 2.0 * log10_R - 2.0 * _LOG10_D10PC + log10_Bnu

    # -> mJy:  1 Jy = 1e-23 erg/cm^2/s/Hz, 1 mJy = 1e-3 Jy
    log10_mJy = log10_Fnu + 23.0 + 3.0
    return log10_mJy


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class AnalyticalModel:
    """Base class for analytical (non-surrogate) light-curve models.

    Subclasses must implement ``compute_log10_lbol_rphot(self, x, t_days)``
    which returns ``(log10_Lbol, log10_Rphot)`` — log10 of bolometric
    luminosity in erg/s and photospheric radius in cm.
    """

    parameter_names: list[str]
    filters: list[str]
    times: Array               # source-frame days

    _nus: Array

    def __init__(self, filters: list[str], times: Array = None,
                 temperature_floor: float | None = None):
        self.filters = []
        self.Filters = []
        self._nus = None
        self.temperature_floor = temperature_floor
        self.add_filter(filters)

        if times is not None:
            self.times = jnp.asarray(times)

    # -- filter management ---------------------------------------------------

    def add_filter(self, filters):
        if isinstance(filters, (str, fiesta_filters.Filter)):
            filters = [filters]
        for filt in filters:
            if isinstance(filt, str):
                F = fiesta_filters.Filter(filt)
            elif isinstance(filt, fiesta_filters.Filter):
                F = filt
            else:
                raise ValueError("Filter must be a name string or Filter object.")
            if F.name not in self.filters:
                self.filters.append(F.name)
                self.Filters.append(F)
        self._build_nu_grid()

    def _build_nu_grid(self):
        """Build a log-spaced frequency grid spanning all loaded filters."""
        if len(self.Filters) == 0:
            return
        nu_min = min(F.nus[0] for F in self.Filters)
        nu_max = max(F.nus[-1] for F in self.Filters)
        self._nus = jnp.logspace(jnp.log10(nu_min * 0.95),
                                 jnp.log10(nu_max * 1.05), 200)

    # -- physics (to be overridden) ------------------------------------------

    def compute_log10_lbol_rphot(self, x: dict[str, Array],
                                 t_days: Array) -> tuple[Array, Array]:
        """Return (log10_L_bol, log10_R_phot) arrays at each time in *t_days*.

        L_bol in erg/s, R_phot in cm.
        """
        raise NotImplementedError

    # -- predict (JIT-compiled) ---------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def predict(self, x: dict[str, Array]) -> tuple[Array, dict[str, Array]]:
        t_days = self.times

        log10_L, log10_R = self.compute_log10_lbol_rphot(x, t_days)

        # Temperature from Stefan-Boltzmann: T = (L / (4 pi R^2 sigma))^{1/4}
        log10_T = 0.25 * (log10_L - _LOG10_4PI - 2.0 * log10_R - _LOG10_SIGMASB)

        # Apply temperature floor if set (branch-free for JIT)
        if self.temperature_floor is not None:
            log10_T_floor = jnp.log10(self.temperature_floor)
            below_floor = log10_T < log10_T_floor
            log10_T = jnp.where(below_floor, log10_T_floor, log10_T)
            # Adjust R so that L = 4*pi*R^2*sigma*T_floor^4
            log10_R_eff = 0.5 * (log10_L - _LOG10_4PI - _LOG10_SIGMASB
                                 - 4.0 * log10_T_floor)
            log10_R = jnp.where(below_floor, log10_R_eff, log10_R)

        # Clamp temperature to [100, 1e6] K
        log10_T = jnp.clip(log10_T, 2.0, 6.0)
        T = jnp.power(10.0, log10_T)

        # Blackbody SED at each time -> magnitudes per filter
        nus = self._nus

        def _sed_at_t(args):
            T_i, log10_R_i = args
            return _log10_blackbody_mJy_at_10pc(T_i, log10_R_i, nus)

        # (n_times, n_nus) of log10(mJy)
        log10_sed = jax.vmap(_sed_at_t)((T, log10_R))
        # Convert to linear mJy: (n_times, n_nus) -> transpose to (n_nus, n_times)
        sed = jnp.power(10.0, log10_sed).T

        # Redshift: shift frequencies only (source-frame times returned)
        z = x.get("redshift", 0.0)
        nus_obs = nus / (1.0 + z)
        sed_obs = sed * (1.0 + z)

        mag_abs = {F.name: F.get_mag(sed_obs, nus_obs) for F in self.Filters}

        dL = x["luminosity_distance"]
        mag_app = {k: mag_app_from_mag_abs(v, dL) for k, v in mag_abs.items()}

        return t_days, mag_app


# ---------------------------------------------------------------------------
# Piro (2021) shock-cooling model
# ---------------------------------------------------------------------------

class ShockCoolingModel(AnalyticalModel):
    """Shock-cooling emission following Piro (2021).

    Parameters (all in ``x`` dict):
        log10_Menv  – log10 envelope mass in solar masses
        log10_Renv  – log10 envelope radius in solar radii
        log10_Ee    – log10 explosion energy in erg
    """

    parameter_names = ["log10_Menv", "log10_Renv", "log10_Ee"]

    _kappa = 0.2  # cm^2/g (electron scattering)

    def __init__(self, filters, times=None):
        if times is None:
            times = jnp.geomspace(1.0 / 24.0, 3.5, 100)
        super().__init__(filters, times)

    def compute_log10_lbol_rphot(self, x, t_days):
        # All heavy computation in log10 space
        log10_Me = x["log10_Menv"] + _LOG10_MSUN
        log10_Re = x["log10_Renv"] + _LOG10_RSUN
        log10_Ee = x["log10_Ee"]
        log10_kappa = jnp.log10(self._kappa)

        # Velocity: ve = sqrt(2 Ee / Me)
        log10_ve = 0.5 * (jnp.log10(2.0) + log10_Ee - log10_Me)

        # Diffusion time: td = sqrt(kappa * Me / (ve * c))
        log10_td = 0.5 * (log10_kappa + log10_Me - log10_ve - _LOG10_CCGS)

        # Convert td and ve to linear (they're moderate: ~1e5 s and ~1e9 cm/s)
        td = jnp.power(10.0, log10_td)
        ve = jnp.power(10.0, log10_ve)

        t = t_days * days_to_seconds
        t = jnp.maximum(t, 1.0)

        # Exponential decay factor: exp(-t*(t + 2*td) / (2*td^2))
        exp_arg = -t * (t + 2.0 * td) / (2.0 * td**2)
        log10_exp = exp_arg * _LOG10E

        # L_early = Ee * Re / (ve * td^2) * exp_factor
        # log10(L_early) = log10_Ee + log10_Re - log10_ve - 2*log10_td + log10_exp
        log10_L_early = log10_Ee + log10_Re - log10_ve - 2.0 * log10_td + log10_exp

        # L_late = Ee * Re / (ve * t^2) * exp_factor
        log10_t = jnp.log10(t)
        log10_L_late = log10_Ee + log10_Re - log10_ve - 2.0 * log10_t + log10_exp

        log10_L = jnp.where(t < td, log10_L_early, log10_L_late)
        log10_L = jnp.maximum(log10_L, 0.0)  # floor at 1 erg/s

        # Photospheric radius: R = Re + ve * min(t, td)
        # Both Re and ve*t are moderate, safe in linear
        Re = jnp.power(10.0, log10_Re)
        R_phot = Re + ve * jnp.minimum(t, td)
        log10_R = jnp.log10(jnp.maximum(R_phot, 1.0))

        return log10_L, log10_R


# ---------------------------------------------------------------------------
# Arnett (1982) Ni56-powered supernova model
# ---------------------------------------------------------------------------

def _gauss_legendre_nodes_weights(n=32):
    """Return n-point Gauss-Legendre nodes/weights on [0, 1]."""
    import numpy as np
    nodes, weights = np.polynomial.legendre.leggauss(n)
    nodes = (nodes + 1.0) / 2.0
    weights = weights / 2.0
    return jnp.array(nodes), jnp.array(weights)


class ArnettModel(AnalyticalModel):
    """Arnett (1982) Ni56/Co56-powered supernova bolometric model.

    Parameters (in ``x`` dict):
        tau_m       – diffusion timescale in days
        log10_mni   – log10 of Ni56 mass in solar masses
        v_phot      – photospheric velocity in units of 1e9 cm/s
        t_0         – (modified variant only) gamma-ray trapping timescale in days
    """

    parameter_names = ["tau_m", "log10_mni", "v_phot"]

    # Ni56 and Co56 decay timescales (days)
    _tau_ni = 8.77
    _tau_co = 111.3

    # log10 of energy release per gram (erg/s/g)
    _log10_eps_ni = jnp.log10(3.90e10)
    _log10_eps_co = jnp.log10(6.78e9)

    def __init__(self, filters, times=None, modified=False):
        self.modified = modified
        if modified:
            self.parameter_names = ["tau_m", "log10_mni", "v_phot", "t_0"]
        else:
            self.parameter_names = list(ArnettModel.parameter_names)
        if times is None:
            times = jnp.geomspace(0.1, 60.0, 100)
        self._gl_nodes, self._gl_weights = _gauss_legendre_nodes_weights()
        super().__init__(filters, times)

    def compute_log10_lbol_rphot(self, x, t_days):
        tau_m = x["tau_m"]                          # days
        log10_Mni = x["log10_mni"] + _LOG10_MSUN    # log10(grams)
        v_phot = x["v_phot"]                        # 1e9 cm/s

        tau_ni = self._tau_ni
        tau_co = self._tau_co

        nodes = self._gl_nodes
        weights = self._gl_weights

        def _log10_lbol_single(t_d):
            y = t_d / tau_m
            z = y * nodes  # (32,)

            # Ni56 integral: I_ni = y * integral of 2*z*exp(-z^2 + z*tau_m/tau_ni)
            A_ni = 2.0 * z * jnp.exp(-z**2 + z * tau_m / tau_ni)
            I_ni = y * jnp.dot(weights, A_ni)

            # Co56 integral
            A_co = 2.0 * z * jnp.exp(-z**2 + z * tau_m / tau_co)
            I_co = y * jnp.dot(weights, A_co)

            # Luminosity in log10:
            # L = M_ni * exp(-(t/tau_m)^2) * (eps_ni * exp(-t/tau_ni) * I_ni
            #                                + eps_co * exp(-t/tau_co) * I_co)
            # The integrals and exp terms are moderate, compute in linear then take log10
            decay_ni = jnp.exp(-(t_d / tau_m)**2 - t_d / tau_ni) * I_ni
            decay_co = jnp.exp(-(t_d / tau_m)**2 - t_d / tau_co) * I_co

            # eps_ni * decay_ni + eps_co * decay_co  (need to combine in linear)
            # Use log10(a + b) = log10(a) + log10(1 + b/a)
            # But eps values are moderate (1e10), so compute partial sums safely
            # Total = 10^log10_eps_ni * decay_ni + 10^log10_eps_co * decay_co
            term_ni = jnp.power(10.0, self._log10_eps_ni) * decay_ni
            term_co = jnp.power(10.0, self._log10_eps_co) * decay_co
            total_heating = term_ni + term_co

            # log10(L) = log10(M_ni) + log10(total_heating)
            log10_L = log10_Mni + jnp.log10(jnp.maximum(total_heating, 1e-30))
            return log10_L

        log10_L = jax.vmap(_log10_lbol_single)(t_days)

        # Modified variant: multiply by trapping factor
        if self.modified:
            t_0 = x["t_0"]
            trap = 1.0 - jnp.exp(-(t_0 / t_days)**2)
            log10_L = log10_L + jnp.log10(jnp.maximum(trap, 1e-30))

        log10_L = jnp.maximum(log10_L, 0.0)

        # Photospheric radius: R = v_phot * 1e9 * t_sec
        t_sec = t_days * days_to_seconds
        # log10(R) = log10(v_phot) + 9 + log10(t_sec)
        log10_R = jnp.log10(jnp.maximum(v_phot, 1e-10)) + 9.0 + jnp.log10(t_sec)

        return log10_L, log10_R


# ---------------------------------------------------------------------------
# Metzger single-zone kilonova model
# ---------------------------------------------------------------------------

class MetzgerModel(AnalyticalModel):
    """Efficient single-zone kilonova model (Metzger 2017).

    Parameters (in ``x`` dict):
        log10_mej     – log10 ejecta mass in solar masses
        log10_vej     – log10 ejecta velocity in units of c
        beta          – velocity power-law index
        log10_kappa_r – log10 opacity in cm^2/g

    The ODE is solved in normalized units to avoid float32 overflow.
    Energy scale: Q_scale = eps_0 * mej (heating rate at 1 day).
    """

    parameter_names = ["log10_mej", "log10_vej", "beta", "log10_kappa_r"]

    _n_internal = 500

    def __init__(self, filters, times=None):
        if times is None:
            times = jnp.geomspace(0.1, 30.0, 100)
        super().__init__(filters, times)

    def compute_log10_lbol_rphot(self, x, t_days):
        log10_mej = x["log10_mej"] + _LOG10_MSUN
        log10_vej = x["log10_vej"] + _LOG10_CCGS
        kappa = jnp.power(10.0, x["log10_kappa_r"])

        vej = jnp.power(10.0, log10_vej)

        t_out = t_days * days_to_seconds
        t_start = t_out[0] * 0.1
        t_end = t_out[-1] * 1.1
        t_int = jnp.linspace(t_start, t_end, self._n_internal)
        dt = t_int[1] - t_int[0]

        # Energy normalization: Q_scale = eps_0 * mej
        # log10(Q_scale) = log10(2e10) + log10_mej
        log10_Q_scale = jnp.log10(2.0e10) + log10_mej
        alpha_heat = 1.3

        # Diffusion timescale: td = sqrt(3 * kappa * mej / (4 pi * vej * c))
        # Compute in log: log10_td = 0.5 * (log10(3*kappa) + log10_mej - log10(4pi) - log10_vej - log10_c)
        log10_td = 0.5 * (jnp.log10(3.0 * kappa) + log10_mej
                          - _LOG10_4PI - log10_vej - _LOG10_CCGS)
        td = jnp.power(10.0, log10_td)

        def _eps_th(t_sec):
            t_d = t_sec / days_to_seconds
            return 0.36 * jnp.exp(-0.56 * t_d) + 0.44 / (1.0 + (t_d / 0.28)**0.62)

        # ODE in normalized units: E_n = E / Q_scale, L_n = L / Q_scale
        # dE_n/dt = t_d^{-alpha} * eps_th  -  E_n/t * min(t/td, 1)  -  E_n/(3t)
        def _scan_step(carry, t_i):
            E_n = carry
            t_d = t_i / days_to_seconds
            Q_n = jnp.power(jnp.maximum(t_d, 1e-4), -alpha_heat) * _eps_th(t_i)

            L_n = E_n / jnp.maximum(t_i, 1.0) * jnp.minimum(t_i / td, 1.0)

            dEndt = Q_n - L_n - E_n / (3.0 * jnp.maximum(t_i, 1.0))
            E_new = jnp.maximum(E_n + dEndt * dt, 0.0)
            return E_new, L_n

        t_d_start = t_start / days_to_seconds
        E0_n = jnp.power(jnp.maximum(t_d_start, 1e-4), -alpha_heat) * t_start * 0.5
        _, L_n_int = jax.lax.scan(_scan_step, E0_n, t_int)

        L_n_out = jnp.interp(t_out, t_int, L_n_int)
        # log10(L) = log10(L_n) + log10(Q_scale)
        log10_L = jnp.log10(jnp.maximum(L_n_out, 1e-30)) + log10_Q_scale
        log10_L = jnp.maximum(log10_L, 0.0)

        log10_R = log10_vej + jnp.log10(t_out)

        return log10_L, log10_R


# ---------------------------------------------------------------------------
# Metzger full multi-shell kilonova model
# ---------------------------------------------------------------------------

class MetzgerFullModel(AnalyticalModel):
    """Full multi-shell kilonova model (Metzger 2017).

    Same parameters as ``MetzgerModel`` but resolves 300 mass shells with a
    velocity profile and shell-dependent opacities/heating.

    The ODE is solved in normalized units (same as MetzgerModel) to avoid
    float32 overflow.

    Parameters (in ``x`` dict):
        log10_mej     – log10 ejecta mass in solar masses
        log10_vej     – log10 ejecta velocity in units of c
        beta          – velocity power-law index
        log10_kappa_r – log10 opacity in cm^2/g
    """

    parameter_names = ["log10_mej", "log10_vej", "beta", "log10_kappa_r"]

    _n_shells = 300
    _n_internal = 500

    def __init__(self, filters, times=None):
        if times is None:
            times = jnp.geomspace(0.1, 30.0, 100)
        super().__init__(filters, times)

    def compute_log10_lbol_rphot(self, x, t_days):
        log10_mej = x["log10_mej"] + _LOG10_MSUN
        log10_vej = x["log10_vej"] + _LOG10_CCGS
        beta_val = x["beta"]
        kappa_base = jnp.power(10.0, x["log10_kappa_r"])

        n_shells = self._n_shells
        vej = jnp.power(10.0, log10_vej)

        t_out = t_days * days_to_seconds
        t_start = t_out[0] * 0.1
        t_end = t_out[-1] * 1.1
        t_int = jnp.linspace(t_start, t_end, self._n_internal)
        dt = t_int[1] - t_int[0]

        # Energy normalization: Q_scale = eps_0 * dm_shell = eps_0 * mej / n_shells
        # log10(Q_scale) = log10(2e10) + log10_mej - log10(n_shells)
        log10_Q_scale = jnp.log10(2.0e10) + log10_mej - jnp.log10(float(n_shells))

        m_frac = jnp.linspace(1.0 / n_shells, 1.0, n_shells)
        v_shells = vej * jnp.power(m_frac, -1.0 / beta_val)
        v_shells = jnp.minimum(v_shells, 0.5 * c_cgs)

        kappa_shells = kappa_base * jnp.ones(n_shells)

        # Diffusion timescale per shell: td = sqrt(3 * kappa * dm / (4pi * v * c))
        # dm = mej / n_shells; compute td in log space then convert
        log10_dm = log10_mej - jnp.log10(float(n_shells))
        log10_td_shells = 0.5 * (jnp.log10(3.0 * kappa_shells) + log10_dm
                                  - _LOG10_4PI - jnp.log10(v_shells) - _LOG10_CCGS)
        td_shells = jnp.power(10.0, log10_td_shells)

        # For the optical depth calculation, we need dm in linear.
        # dm = 10^log10_dm; may overflow float32 if mej is large.
        # But dm/r^2 is what we need, and we can compute log10(dm) - 2*log10(r).
        # For simplicity, use a normalized dm_n = 1 and scale optical depth at the end.
        # tau = kappa * dm / (4pi * r^2) = 10^(log10_kappa + log10_dm - log10_4pi - 2*log10_r)
        # We'll compute this in log space.
        alpha_heat = 1.3

        def _eps_th(t_sec):
            t_d = t_sec / days_to_seconds
            return 0.36 * jnp.exp(-0.56 * t_d) + 0.44 / (1.0 + (t_d / 0.28)**0.62)

        # ODE in normalized units: E_n = E_shell / Q_scale
        def _scan_step(carry, t_i):
            E_n_shells = carry  # (n_shells,)
            t_d = t_i / days_to_seconds

            Q_n = jnp.power(jnp.maximum(t_d, 1e-4), -alpha_heat) * _eps_th(t_i)

            L_n_shells = E_n_shells / jnp.maximum(t_i, 1.0) * jnp.minimum(t_i / td_shells, 1.0)
            pdv = E_n_shells / (3.0 * jnp.maximum(t_i, 1.0))

            dEndt = Q_n - L_n_shells - pdv
            E_new = jnp.maximum(E_n_shells + dEndt * dt, 0.0)

            L_n_total = jnp.sum(L_n_shells)

            # Photospheric radius from optical depth (computed in log space)
            r_shells = v_shells * t_i
            log10_r = jnp.log10(jnp.maximum(r_shells, 1.0))
            log10_tau_per_shell = (jnp.log10(kappa_shells) + log10_dm
                                   - _LOG10_4PI - 2.0 * log10_r)
            tau_per_shell = jnp.power(10.0, log10_tau_per_shell)
            tau_shells = jnp.cumsum(tau_per_shell[::-1])[::-1]

            is_above = tau_shells > 2.0 / 3.0
            weights_ph = jnp.where(is_above, 1.0, 0.0)
            total_w = jnp.sum(weights_ph) + 1e-30
            R_ph = jnp.sum(weights_ph * r_shells) / total_w
            R_ph = jnp.maximum(R_ph, r_shells[-1])

            return E_new, (L_n_total, R_ph)

        t_d_start = t_start / days_to_seconds
        E0_n = jnp.ones(n_shells) * jnp.power(
            jnp.maximum(t_d_start, 1e-4), -alpha_heat
        ) * t_start * 0.5

        _, (L_n_int, R_int) = jax.lax.scan(_scan_step, E0_n, t_int)

        L_n_out = jnp.interp(t_out, t_int, L_n_int)
        # Total L = L_n * Q_scale * n_shells (sum of n_shells each contributing Q_scale * L_n_per_shell)
        # But L_n_total already sums over shells, so L = L_n_total * Q_scale
        log10_L = jnp.log10(jnp.maximum(L_n_out, 1e-30)) + log10_Q_scale
        log10_L = jnp.maximum(log10_L, 0.0)

        R_phot = jnp.interp(t_out, t_int, R_int)
        log10_R = jnp.log10(jnp.maximum(R_phot, 1.0))

        return log10_L, log10_R


# ---------------------------------------------------------------------------
# Magnetar spin-down luminosity helper
# ---------------------------------------------------------------------------

def _magnetar_luminosity(t_sec, log10_p0, log10_bp, mass_ns, theta_pb):
    """Magnetar spin-down luminosity (dipole radiation).

    All intermediate quantities computed in log10 space to avoid float32
    overflow (E_rot ~ 1e52 exceeds float32 max).

    Parameters
    ----------
    t_sec : array – time in seconds
    log10_p0 : scalar – log10 of initial spin period in milliseconds
    log10_bp : scalar – log10 of polar B-field in units of 1e14 G
    mass_ns : scalar – neutron star mass in solar masses
    theta_pb : scalar – angle between spin and B-field axes in radians

    Returns
    -------
    log10_L_mag : array – log10 of magnetar luminosity in erg/s
    """
    # log10(E_rot) = log10(2.6e52) + 1.5*log10(M/1.4) - 2*log10(p0_ms)
    log10_mass_ratio = jnp.log10(mass_ns / 1.4)
    log10_E_rot = 52.0 + jnp.log10(2.6) + 1.5 * log10_mass_ratio - 2.0 * log10_p0

    # tau_p = 1.3e5 * Bp14^{-2} * P_ms^2 * (M/1.4)^{3/2} * sin(theta)^{-2}
    # Compute in log10 to avoid overflow
    log10_tau_p = (5.0 + jnp.log10(1.3)
                   - 2.0 * log10_bp + 2.0 * log10_p0
                   + 1.5 * log10_mass_ratio
                   - 2.0 * jnp.log10(jnp.sin(theta_pb)))
    tau_p = jnp.power(10.0, log10_tau_p)

    # L_mag = E_rot / tau_p / (1 + t/tau_p)^2
    log10_L_mag = (log10_E_rot - log10_tau_p
                   - 2.0 * jnp.log10(1.0 + t_sec / tau_p))
    return log10_L_mag


# ---------------------------------------------------------------------------
# One-component kilonova model (Kasen+17 style diffusion integral)
# ---------------------------------------------------------------------------

class OneComponentKilonovaModel(AnalyticalModel):
    """Single-component kilonova with diffusion-integral heating.

    Different from ``MetzgerModel`` (energy-balance ODE): this model computes
    L_bol via a cumulative heating integral with diffusion damping, reformulated
    as a stable first-order ODE to avoid ``exp(t^2/td^2)`` overflow.

    Parameters (in ``x`` dict):
        log10_mej   – log10 ejecta mass in solar masses
        log10_vej   – log10 ejecta velocity in units of c
        log10_kappa – log10 gray opacity in cm^2/g
    """

    parameter_names = ["log10_mej", "log10_vej", "log10_kappa"]

    _n_internal = 500

    def __init__(self, filters, times=None, temperature_floor=4000.0):
        if times is None:
            times = jnp.geomspace(0.1, 30.0, 100)
        super().__init__(filters, times, temperature_floor=temperature_floor)

    def compute_log10_lbol_rphot(self, x, t_days):
        log10_mej_g = x["log10_mej"] + _LOG10_MSUN
        log10_vej_cms = x["log10_vej"] + _LOG10_CCGS
        log10_kappa = x["log10_kappa"]

        kappa = jnp.power(10.0, log10_kappa)

        t_out = t_days * days_to_seconds
        t_start = jnp.maximum(t_out[0] * 0.1, 1.0)
        t_end = t_out[-1] * 1.1
        t_int = jnp.linspace(t_start, t_end, self._n_internal)
        dt = t_int[1] - t_int[0]

        # Diffusion timescale in log10 space to avoid overflow:
        # td = sqrt(2 * kappa * mej / (13.7 * vej * c))
        log10_td = 0.5 * (jnp.log10(2.0) + log10_kappa + log10_mej_g
                          - jnp.log10(13.7) - log10_vej_cms - _LOG10_CCGS)
        td = jnp.power(10.0, log10_td)

        # Normalization: Q_scale = 4e18 * mej (peak heating rate)
        # log10(Q_scale) = log10(4e18) + log10_mej_g  (~50, overflows float32)
        log10_Q_scale = jnp.log10(4.0e18) + log10_mej_g

        # Thermalisation efficiency (simplified Barnes+16)
        def _e_th(t_sec):
            t_d = t_sec / days_to_seconds
            return (0.36 * jnp.exp(-0.56 * t_d)
                    + 0.44 / (1.0 + (t_d / 0.28)**0.62))

        # Normalized heating: L_in_n(t) = (0.5 - arctan((t-1.3)/0.11)/pi)^1.3
        # so that L_in = Q_scale * L_in_n
        t0_heat = 1.3   # seconds
        sig_heat = 0.11  # seconds

        def _l_in_n(t_sec):
            return jnp.power(
                0.5 - jnp.arctan((t_sec - t0_heat) / sig_heat) / jnp.pi, 1.3)

        # Stable ODE in normalized units:
        #   U_n(t) = td * L_bol_n(t),  where L_bol = Q_scale * L_bol_n
        #   dU_n/dt = L_in_n * e_th * (t/td)  -  2*(t/td^2) * U_n
        #   L_bol_n = U_n / td

        def _scan_step(U_n, t_i):
            heating = _l_in_n(t_i) * _e_th(t_i) * (t_i / td)
            loss = 2.0 * t_i / td**2 * U_n
            dU_n_dt = heating - loss
            U_n_new = jnp.maximum(U_n + dU_n_dt * dt, 0.0)
            L_n = jnp.maximum(U_n_new / td, 0.0)
            return U_n_new, L_n

        U0_n = _l_in_n(t_start) * _e_th(t_start) * t_start / td * dt
        _, L_n_int = jax.lax.scan(_scan_step, U0_n, t_int)

        L_n_out = jnp.interp(t_out, t_int, L_n_int)
        log10_L = jnp.log10(jnp.maximum(L_n_out, 1e-30)) + log10_Q_scale
        log10_L = jnp.maximum(log10_L, 0.0)

        # Photospheric radius: R = vej * t
        log10_R = log10_vej_cms + jnp.log10(t_out)

        return log10_L, log10_R


# ---------------------------------------------------------------------------
# Magnetar-boosted kilonova (Metzger + magnetar spin-down)
# ---------------------------------------------------------------------------

class MagnetarBoostedKilonovaModel(AnalyticalModel):
    """Single-zone kilonova with magnetar spin-down heating injection.

    Extends the MetzgerModel energy-balance ODE with an additional magnetar
    luminosity term.

    Parameters (in ``x`` dict):
        log10_mej     – log10 ejecta mass in solar masses
        log10_vej     – log10 ejecta velocity in units of c
        beta          – velocity power-law index
        log10_kappa_r – log10 opacity in cm^2/g
        log10_p0      – log10 initial spin period in ms
        log10_bp      – log10 polar B-field in 1e14 G
        mass_ns       – neutron star mass in solar masses
        theta_pb      – angle between spin and B-field in radians
    """

    parameter_names = ["log10_mej", "log10_vej", "beta", "log10_kappa_r",
                       "log10_p0", "log10_bp", "mass_ns", "theta_pb"]

    _n_internal = 500

    def __init__(self, filters, times=None):
        if times is None:
            times = jnp.geomspace(0.1, 30.0, 100)
        super().__init__(filters, times)

    def compute_log10_lbol_rphot(self, x, t_days):
        log10_mej = x["log10_mej"] + _LOG10_MSUN
        log10_vej = x["log10_vej"] + _LOG10_CCGS
        kappa = jnp.power(10.0, x["log10_kappa_r"])

        vej = jnp.power(10.0, log10_vej)

        t_out = t_days * days_to_seconds
        t_start = t_out[0] * 0.1
        t_end = t_out[-1] * 1.1
        t_int = jnp.linspace(t_start, t_end, self._n_internal)
        dt = t_int[1] - t_int[0]

        # Energy normalization: Q_scale = eps_0 * mej
        log10_Q_scale = jnp.log10(2.0e10) + log10_mej
        alpha_heat = 1.3

        # Diffusion timescale
        log10_td = 0.5 * (jnp.log10(3.0 * kappa) + log10_mej
                          - _LOG10_4PI - log10_vej - _LOG10_CCGS)
        td = jnp.power(10.0, log10_td)

        # Magnetar luminosity on internal time grid
        log10_L_mag = _magnetar_luminosity(
            t_int, x["log10_p0"], x["log10_bp"], x["mass_ns"], x["theta_pb"])

        def _eps_th(t_sec):
            t_d = t_sec / days_to_seconds
            return 0.36 * jnp.exp(-0.56 * t_d) + 0.44 / (1.0 + (t_d / 0.28)**0.62)

        # ODE: dE_n/dt = Q_rp_n + L_mag_n * e_th  -  L_n  -  E_n/(3t)
        def _scan_step(carry, inputs):
            E_n = carry
            t_i, log10_L_mag_i = inputs
            t_d = t_i / days_to_seconds
            eth = _eps_th(t_i)

            # r-process heating (normalized)
            Q_n = jnp.power(jnp.maximum(t_d, 1e-4), -alpha_heat) * eth

            # Magnetar heating (normalized): L_mag_n = L_mag / Q_scale
            # Compute in log10 to avoid overflow (both ~1e47-1e52)
            L_mag_n = jnp.power(10.0, log10_L_mag_i - log10_Q_scale) * eth

            L_n = E_n / jnp.maximum(t_i, 1.0) * jnp.minimum(t_i / td, 1.0)
            pdv = E_n / (3.0 * jnp.maximum(t_i, 1.0))

            dEndt = Q_n + L_mag_n - L_n - pdv
            E_new = jnp.maximum(E_n + dEndt * dt, 0.0)
            return E_new, L_n

        t_d_start = t_start / days_to_seconds
        E0_n = jnp.power(jnp.maximum(t_d_start, 1e-4), -alpha_heat) * t_start * 0.5

        _, L_n_int = jax.lax.scan(_scan_step, E0_n, (t_int, log10_L_mag))

        L_n_out = jnp.interp(t_out, t_int, L_n_int)
        log10_L = jnp.log10(jnp.maximum(L_n_out, 1e-30)) + log10_Q_scale
        log10_L = jnp.maximum(log10_L, 0.0)

        log10_R = log10_vej + jnp.log10(t_out)

        return log10_L, log10_R


# ---------------------------------------------------------------------------
# Shocked cocoon model
# ---------------------------------------------------------------------------

class ShockedCocoonModel(AnalyticalModel):
    """Analytical jet cocoon cooling model.

    Fully algebraic (no ODE) — power-law luminosity decay with diffusion
    timescale.  Based on the shocked cocoon model from Redback.

    Parameters (in ``x`` dict):
        log10_mej          – log10 ejecta mass in solar masses
        log10_vej          – log10 ejecta velocity in units of c
        eta                – slope for ejecta density profile
        log10_tshock       – log10 shock time in seconds
        shocked_fraction   – fraction of ejecta mass shocked
        cos_theta_cocoon   – cosine of cocoon opening angle
        log10_kappa        – log10 gray opacity in cm^2/g
    """

    parameter_names = ["log10_mej", "log10_vej", "eta", "log10_tshock",
                       "shocked_fraction", "cos_theta_cocoon", "log10_kappa"]

    def __init__(self, filters, times=None):
        if times is None:
            times = jnp.geomspace(0.01, 30.0, 100)
        super().__init__(filters, times)

    def compute_log10_lbol_rphot(self, x, t_days):
        log10_mej = x["log10_mej"]                     # solar masses
        log10_vej = x["log10_vej"]                     # units of c
        eta = x["eta"]
        log10_tshock = x["log10_tshock"]               # seconds
        f_sh = x["shocked_fraction"]
        cos_theta = x["cos_theta_cocoon"]
        log10_kappa = x["log10_kappa"]

        theta = jnp.arccos(cos_theta)
        # log10 of CGS quantities
        log10_vej_cms = log10_vej + _LOG10_CCGS        # cm/s
        log10_rshock = log10_tshock + _LOG10_CCGS      # cm
        log10_Msh_g = jnp.log10(f_sh) + log10_mej + _LOG10_MSUN  # grams

        vej_cms = jnp.power(10.0, log10_vej_cms)      # moderate: ~1e9-1e10

        # Diffusion timescale (days):
        # tau_diff = sqrt(Msun * kappa * f_sh * mej / (4pi * c * vej_cms)) / day_to_s
        log10_tau_diff_s = 0.5 * (_LOG10_MSUN + log10_kappa
                                  + jnp.log10(f_sh) + log10_mej
                                  - _LOG10_4PI - _LOG10_CCGS - log10_vej_cms)
        tau_diff = jnp.power(10.0, log10_tau_diff_s) / days_to_seconds  # days
        log10_tau_diff = jnp.log10(tau_diff)

        # Transition time (days): t_thin = sqrt(c/vej) * tau_diff
        t_thin = jnp.sqrt(c_cgs / vej_cms) * tau_diff

        # Luminosity scale in log10:
        # L_scale = (theta^2/2)^(1/3) * M_sh_g * vej_cms * rshock / (tau_diff_s)^2
        log10_theta_factor = (1.0 / 3.0) * jnp.log10(theta**2 / 2.0)
        log10_L_scale = (log10_theta_factor + log10_Msh_g + log10_vej_cms
                         + log10_rshock - 2.0 * log10_tau_diff_s)

        # Bolometric luminosity in log10:
        # L_bol = L_scale * (t/tau_diff)^(-4/(eta+2)) * (1+tanh(t_thin-t))/2
        power_term = -4.0 / (eta + 2.0) * jnp.log10(jnp.maximum(t_days / tau_diff, 1e-10))
        tanh_term = jnp.log10(jnp.maximum(
            (1.0 + jnp.tanh(t_thin - t_days)) / 2.0, 1e-30))

        log10_L = log10_L_scale + power_term + tanh_term
        log10_L = jnp.maximum(log10_L, 0.0)

        # Photospheric velocity and radius (in log10)
        # v_phot = vej * (t/t_thin)^(-2/(eta+3))
        log10_v_phot = (log10_vej_cms
                        + (-2.0 / (eta + 3.0))
                        * jnp.log10(jnp.maximum(t_days / t_thin, 1e-10)))
        # R_phot = v_phot * t_days * day_to_s
        log10_R = log10_v_phot + jnp.log10(t_days) + _LOG10_DAYS2SEC

        return log10_L, log10_R


# ---------------------------------------------------------------------------
# Evolving blackbody model
# ---------------------------------------------------------------------------

class EvolvingBlackbodyModel(AnalyticalModel):
    """Phenomenological model with piecewise power-law T and R evolution.

    Model-agnostic — useful for fast empirical fitting of any thermal
    transient.  Based on the evolving_blackbody model from Redback.

    Parameters (in ``x`` dict):
        log10_temperature_0   – log10 initial temperature (K) at reference_time
        log10_radius_0        – log10 initial radius (cm) at reference_time
        temp_rise_index       – T rise power-law index for t <= temp_peak_time
        temp_decline_index    – T decline power-law index for t > temp_peak_time
        temp_peak_time        – time (days) when temperature peaks
        radius_rise_index     – R rise power-law index for t <= radius_peak_time
        radius_decline_index  – R decline power-law index for t > radius_peak_time
        radius_peak_time      – time (days) when radius peaks
    """

    parameter_names = [
        "log10_temperature_0", "log10_radius_0",
        "temp_rise_index", "temp_decline_index", "temp_peak_time",
        "radius_rise_index", "radius_decline_index", "radius_peak_time",
    ]

    def __init__(self, filters, times=None, reference_time=1.0):
        self.reference_time = reference_time
        if times is None:
            times = jnp.geomspace(0.1, 30.0, 100)
        super().__init__(filters, times)

    def compute_log10_lbol_rphot(self, x, t_days):
        T0 = jnp.power(10.0, x["log10_temperature_0"])
        R0 = jnp.power(10.0, x["log10_radius_0"])
        a_T_rise = x["temp_rise_index"]
        a_T_dec = x["temp_decline_index"]
        t_pk_T = x["temp_peak_time"]
        a_R_rise = x["radius_rise_index"]
        a_R_dec = x["radius_decline_index"]
        t_pk_R = x["radius_peak_time"]
        t_ref = self.reference_time

        # Temperature evolution (piecewise power-law)
        T_peak = T0 * jnp.power(t_pk_T / t_ref, a_T_rise)
        T = jnp.where(
            t_days <= t_pk_T,
            T0 * jnp.power(t_days / t_ref, a_T_rise),
            T_peak * jnp.power(t_days / t_pk_T, -a_T_dec),
        )

        # Radius evolution (piecewise power-law)
        R_peak = R0 * jnp.power(t_pk_R / t_ref, a_R_rise)
        R = jnp.where(
            t_days <= t_pk_R,
            R0 * jnp.power(t_days / t_ref, a_R_rise),
            R_peak * jnp.power(t_days / t_pk_R, -a_R_dec),
        )

        # L = 4 * pi * R^2 * sigma * T^4
        log10_L = (_LOG10_4PI + 2.0 * jnp.log10(jnp.maximum(R, 1.0))
                   + _LOG10_SIGMASB + 4.0 * jnp.log10(jnp.maximum(T, 1.0)))
        log10_R = jnp.log10(jnp.maximum(R, 1.0))

        return log10_L, log10_R


# ---------------------------------------------------------------------------
# Shared helpers: Arnett (1982) diffusion ODE
# ---------------------------------------------------------------------------

def _compute_diffusion_constants(log10_kappa, log10_kappa_gamma,
                                 log10_mej_g, log10_vej_kms):
    """Compute diffusion timescale and trapping coefficient in log10 space.

    Uses Arnett (1982) conventions with CGS inputs where velocity is in km/s.

    Returns
    -------
    log10_td_days : scalar – log10 of diffusion timescale in days
    log10_A_trap_days2 : scalar – log10 of trapping coefficient in days^2
    """
    # tau_diff = sqrt(2 * kappa * mej_g / (13.7 * c_cgs * vej_cms))
    # where mej_g is already in grams and vej_cms = 10^(log10_vej_kms) * km_cgs
    log10_td_sec = 0.5 * (jnp.log10(2.0) + log10_kappa + log10_mej_g
                          - jnp.log10(13.7) - _LOG10_CCGS
                          - log10_vej_kms - _LOG10_KM_CGS)
    log10_td_days = log10_td_sec - _LOG10_DAYS2SEC

    # A_trap = 3 * kappa_gamma * mej_g / (4*pi * vej_cms^2)
    log10_A_sec2 = (jnp.log10(3.0) + log10_kappa_gamma + log10_mej_g
                    - _LOG10_4PI - 2.0 * (log10_vej_kms + _LOG10_KM_CGS))
    log10_A_trap_days2 = log10_A_sec2 - 2.0 * _LOG10_DAYS2SEC

    return log10_td_days, log10_A_trap_days2


def _arnett_diffusion_ode(log10_L_engine, t_days_grid,
                          log10_td_days, log10_A_trap_days2,
                          log10_L_scale):
    """Arnett (1982) diffusion integral via stable first-order ODE.

    Solves dV/dt = L_engine(t)*t - 2*t/td^2 * V  using ``jax.lax.scan``,
    then L_obs = 2/td^2 * V * (1 - exp(-A_trap/t^2)).

    All luminosity values are normalized by ``10^log10_L_scale`` to keep
    float32-safe O(1) intermediates.

    Parameters
    ----------
    log10_L_engine : 1-D array – log10 of engine luminosity (erg/s)
    t_days_grid : 1-D array – time grid in days (must be uniformly spaced)
    log10_td_days : scalar – log10 of diffusion timescale in days
    log10_A_trap_days2 : scalar – log10 of trapping coefficient in days^2
    log10_L_scale : scalar – normalization scale (log10 erg/s)

    Returns
    -------
    log10_L_obs : 1-D array – log10 of observed luminosity (erg/s)
    """
    td = jnp.power(10.0, log10_td_days)
    A_trap = jnp.power(10.0, log10_A_trap_days2)
    dt = t_days_grid[1] - t_days_grid[0]

    # Normalized engine luminosity
    L_engine_n = jnp.power(10.0, log10_L_engine - log10_L_scale)

    def _scan_step(V_n, inputs):
        L_n_i, t_i = inputs
        source = L_n_i * t_i
        loss = 2.0 * t_i / td**2 * V_n
        V_n_new = jnp.maximum(V_n + (source - loss) * dt, 0.0)
        # Observed luminosity (normalized)
        trap_factor = -jnp.expm1(-A_trap / jnp.maximum(t_i**2, 1e-30))
        L_obs_n = 2.0 / td**2 * V_n_new * trap_factor
        return V_n_new, L_obs_n

    V0 = L_engine_n[0] * t_days_grid[0] * dt
    _, L_obs_n = jax.lax.scan(_scan_step, V0, (L_engine_n, t_days_grid))

    log10_L_obs = jnp.log10(jnp.maximum(L_obs_n, 1e-30)) + log10_L_scale
    return log10_L_obs


# ---------------------------------------------------------------------------
# TDE Analytical Model — t^{-5/3} fallback with ejecta diffusion
# ---------------------------------------------------------------------------

class TDEAnalyticalModel(AnalyticalModel):
    """TDE analytical model with t^{-5/3} fallback + Arnett diffusion.

    Parameters (in ``x`` dict):
        log10_l0         – log10 of luminosity at 1 second (erg/s)
        t_0_turn         – turn-on time in days
        log10_mej        – log10 of ejecta mass in solar masses
        log10_vej        – log10 of ejecta velocity in km/s
        log10_kappa      – log10 of opacity (cm^2/g)
        log10_kappa_gamma – log10 of gamma-ray opacity (cm^2/g)
    """

    parameter_names = ["log10_l0", "t_0_turn", "log10_mej", "log10_vej",
                       "log10_kappa", "log10_kappa_gamma"]

    _n_internal = 500

    def __init__(self, filters, times=None, temperature_floor=None):
        if times is None:
            times = jnp.geomspace(0.1, 200.0, 100)
        super().__init__(filters, times, temperature_floor=temperature_floor)

    def compute_log10_lbol_rphot(self, x, t_days):
        log10_l0 = x["log10_l0"]
        t_0_turn = x["t_0_turn"]
        log10_mej_g = x["log10_mej"] + _LOG10_MSUN
        log10_vej_kms = x["log10_vej"]
        log10_kappa = x["log10_kappa"]
        log10_kappa_gamma = x["log10_kappa_gamma"]

        # Dense internal time grid in days
        t_start = jnp.maximum(t_days[0] * 0.1, 0.01)
        t_end = t_days[-1] * 1.1
        t_int = jnp.linspace(t_start, t_end, self._n_internal)

        # Engine: L(t) = l0 / (max(t, t0_turn) * 86400)^{5/3}
        t_eff = jnp.maximum(t_int, t_0_turn)
        log10_t_sec = jnp.log10(t_eff) + _LOG10_DAYS2SEC
        log10_L_engine = log10_l0 - (5.0 / 3.0) * log10_t_sec

        # Diffusion constants
        log10_td, log10_A_trap = _compute_diffusion_constants(
            log10_kappa, log10_kappa_gamma, log10_mej_g, log10_vej_kms)

        # Scale for normalization
        log10_L_scale = jnp.maximum(jnp.max(log10_L_engine), 30.0)

        # Solve diffusion ODE
        log10_L_int = _arnett_diffusion_ode(
            log10_L_engine, t_int, log10_td, log10_A_trap, log10_L_scale)

        # Interpolate to output times
        L_int = jnp.power(10.0, log10_L_int - log10_L_scale)
        L_out = jnp.interp(t_days, t_int, L_int)
        log10_L = jnp.log10(jnp.maximum(L_out, 1e-30)) + log10_L_scale
        log10_L = jnp.maximum(log10_L, 0.0)

        # Photospheric radius: R = vej * t
        log10_vej_cms = log10_vej_kms + _LOG10_KM_CGS
        t_sec = t_days * days_to_seconds
        log10_R = log10_vej_cms + jnp.log10(t_sec)

        return log10_L, log10_R


# ---------------------------------------------------------------------------
# NickelCobalt Model — Ni56/Co56 decay with Arnett diffusion
# ---------------------------------------------------------------------------

class NickelCobaltModel(AnalyticalModel):
    """Ni56/Co56 radioactive decay with Arnett (1982) diffusion.

    Parameters (in ``x`` dict):
        f_nickel          – fraction of ejecta mass in Ni56
        log10_mej         – log10 of ejecta mass in solar masses
        log10_vej         – log10 of ejecta velocity in km/s
        log10_kappa       – log10 of opacity (cm^2/g)
        log10_kappa_gamma – log10 of gamma-ray opacity (cm^2/g)
    """

    parameter_names = ["f_nickel", "log10_mej", "log10_vej",
                       "log10_kappa", "log10_kappa_gamma"]

    _n_internal = 500

    # Decay parameters
    _ni56_life = 8.8    # days
    _co56_life = 111.3  # days
    # Luminosity per solar mass of Ni56 (erg/s/Msun):
    #   ni56: 6.45e43, co56: 1.45e43
    # Compute log10 without creating the large float (exceeds float32 max)
    _log10_ni56_lum = 43.0 + jnp.log10(6.45)   # 43.8096
    _log10_co56_lum = 43.0 + jnp.log10(1.45)   # 43.1614

    def __init__(self, filters, times=None, temperature_floor=None):
        if times is None:
            times = jnp.geomspace(0.1, 150.0, 100)
        super().__init__(filters, times, temperature_floor=temperature_floor)

    def compute_log10_lbol_rphot(self, x, t_days):
        f_nickel = x["f_nickel"]
        log10_mej_g = x["log10_mej"] + _LOG10_MSUN
        log10_vej_kms = x["log10_vej"]
        log10_kappa = x["log10_kappa"]
        log10_kappa_gamma = x["log10_kappa_gamma"]

        # Nickel mass in log10(solar masses) — the 6.45e43/1.45e43 constants
        # are luminosity per solar mass of Ni56
        log10_mni_solar = jnp.log10(jnp.maximum(f_nickel, 1e-10)) + x["log10_mej"]

        # Dense internal time grid
        t_start = jnp.maximum(t_days[0] * 0.1, 0.01)
        t_end = t_days[-1] * 1.1
        t_int = jnp.linspace(t_start, t_end, self._n_internal)

        # Engine luminosity via log-sum-exp for float32 safety:
        # L(t) = M_ni * (6.45e43 * exp(-t/8.8) + 1.45e43 * exp(-t/111.3))
        # log10(L) = log10(M_ni) + log10(6.45e43*exp(-t/8.8) + 1.45e43*exp(-t/111.3))
        log10_a = self._log10_ni56_lum + (-t_int / self._ni56_life) * _LOG10E
        log10_b = self._log10_co56_lum + (-t_int / self._co56_life) * _LOG10E
        log10_max = jnp.maximum(log10_a, log10_b)
        log10_sum = log10_max + jnp.log10(
            jnp.power(10.0, log10_a - log10_max)
            + jnp.power(10.0, log10_b - log10_max))
        log10_L_engine = log10_mni_solar + log10_sum

        # Diffusion constants
        log10_td, log10_A_trap = _compute_diffusion_constants(
            log10_kappa, log10_kappa_gamma, log10_mej_g, log10_vej_kms)

        log10_L_scale = jnp.maximum(jnp.max(log10_L_engine), 30.0)

        # Solve diffusion ODE
        log10_L_int = _arnett_diffusion_ode(
            log10_L_engine, t_int, log10_td, log10_A_trap, log10_L_scale)

        # Interpolate to output times
        L_int = jnp.power(10.0, log10_L_int - log10_L_scale)
        L_out = jnp.interp(t_days, t_int, L_int)
        log10_L = jnp.log10(jnp.maximum(L_out, 1e-30)) + log10_L_scale
        log10_L = jnp.maximum(log10_L, 0.0)

        # Photospheric radius: R = vej * t
        log10_vej_cms = log10_vej_kms + _LOG10_KM_CGS
        t_sec = t_days * days_to_seconds
        log10_R = log10_vej_cms + jnp.log10(t_sec)

        return log10_L, log10_R


# ---------------------------------------------------------------------------
# Magnetar-Powered Supernova Model
# ---------------------------------------------------------------------------

class MagnetarPoweredSNModel(AnalyticalModel):
    """Magnetar spin-down powered supernova with Arnett (1982) diffusion.

    Parameters (in ``x`` dict):
        log10_p0          – log10 initial spin period in ms
        log10_bp          – log10 polar B-field in 1e14 G
        mass_ns           – neutron star mass in solar masses
        theta_pb          – angle between spin and B-field in radians
        log10_mej         – log10 of ejecta mass in solar masses
        log10_vej         – log10 of ejecta velocity in km/s
        log10_kappa       – log10 of opacity (cm^2/g)
        log10_kappa_gamma – log10 of gamma-ray opacity (cm^2/g)
    """

    parameter_names = ["log10_p0", "log10_bp", "mass_ns", "theta_pb",
                       "log10_mej", "log10_vej", "log10_kappa",
                       "log10_kappa_gamma"]

    _n_internal = 500

    def __init__(self, filters, times=None, temperature_floor=None):
        if times is None:
            times = jnp.geomspace(0.1, 200.0, 100)
        super().__init__(filters, times, temperature_floor=temperature_floor)

    def compute_log10_lbol_rphot(self, x, t_days):
        log10_mej_g = x["log10_mej"] + _LOG10_MSUN
        log10_vej_kms = x["log10_vej"]
        log10_kappa = x["log10_kappa"]
        log10_kappa_gamma = x["log10_kappa_gamma"]

        # Dense internal time grid
        t_start = jnp.maximum(t_days[0] * 0.1, 0.01)
        t_end = t_days[-1] * 1.1
        t_int = jnp.linspace(t_start, t_end, self._n_internal)
        t_int_sec = t_int * days_to_seconds

        # Engine: magnetar spin-down luminosity (already in log10 erg/s)
        log10_L_engine = _magnetar_luminosity(
            t_int_sec, x["log10_p0"], x["log10_bp"],
            x["mass_ns"], x["theta_pb"])

        # Diffusion constants
        log10_td, log10_A_trap = _compute_diffusion_constants(
            log10_kappa, log10_kappa_gamma, log10_mej_g, log10_vej_kms)

        log10_L_scale = jnp.maximum(jnp.max(log10_L_engine), 30.0)

        # Solve diffusion ODE
        log10_L_int = _arnett_diffusion_ode(
            log10_L_engine, t_int, log10_td, log10_A_trap, log10_L_scale)

        # Interpolate to output times
        L_int = jnp.power(10.0, log10_L_int - log10_L_scale)
        L_out = jnp.interp(t_days, t_int, L_int)
        log10_L = jnp.log10(jnp.maximum(L_out, 1e-30)) + log10_L_scale
        log10_L = jnp.maximum(log10_L, 0.0)

        # Photospheric radius: R = vej * t
        log10_vej_cms = log10_vej_kms + _LOG10_KM_CGS
        t_sec = t_days * days_to_seconds
        log10_R = log10_vej_cms + jnp.log10(t_sec)

        return log10_L, log10_R


# ---------------------------------------------------------------------------
# CSM Interaction Model — Chevalier self-similar shock
# ---------------------------------------------------------------------------

class CSMInteractionModel(AnalyticalModel):
    """Circumstellar medium interaction model (Chevalier 1982).

    Forward + reverse shock luminosity from Chevalier self-similar solution,
    with optional CSM diffusion.

    Parameters (in ``x`` dict):
        log10_mej       – log10 of ejecta mass in solar masses
        log10_csm_mass  – log10 of CSM mass in solar masses
        log10_vej       – log10 of ejecta velocity in km/s
        eta             – CSM density profile exponent
        log10_rho       – log10 of CSM density amplitude (g/cm^{eta+3})
        log10_kappa     – log10 of opacity (cm^2/g)
        log10_r0        – log10 of CSM inner radius in AU

    Constructor kwargs:
        nn          – ejecta power-law index (default 12)
        delta       – inner density exponent (default 1)
        efficiency  – kinetic-to-luminosity conversion (default 0.5)
    """

    parameter_names = ["log10_mej", "log10_csm_mass", "log10_vej", "eta",
                       "log10_rho", "log10_kappa", "log10_r0"]

    _n_internal = 500

    def __init__(self, filters, times=None, nn=12, delta=1, efficiency=0.5,
                 temperature_floor=None):
        self.nn = nn
        self.delta = delta
        self.efficiency = efficiency

        # Load CSM table and pre-interpolate along nn axis
        self._load_csm_table(nn)

        if times is None:
            times = jnp.geomspace(0.1, 300.0, 100)
        super().__init__(filters, times, temperature_floor=temperature_floor)

    def _load_csm_table(self, nn_val):
        """Load CSM coefficient table and pre-interpolate for fixed nn."""
        import numpy as np
        import os

        table_path = os.path.join(os.path.dirname(__file__),
                                  "tables", "csm_table.txt")
        data = np.loadtxt(table_path, delimiter=',')
        # Columns: eta_col(10 unique), nn_col(30 unique), AA, Bf, Br
        eta_col = data[:, 0]
        nn_col = data[:, 1]

        # Reshape: 10 eta values x 30 nn values
        n_eta = len(np.unique(eta_col))
        n_nn = len(np.unique(nn_col))
        AA_grid = data[:, 2].reshape(n_eta, n_nn)  # (10, 30)
        Bf_grid = data[:, 3].reshape(n_eta, n_nn)
        Br_grid = data[:, 4].reshape(n_eta, n_nn)

        eta_vals = np.unique(eta_col)  # 10 values
        nn_vals = np.unique(nn_col)    # 30 values

        # Interpolate along nn axis for fixed nn_val -> 1D arrays of eta
        nn_val_clipped = np.clip(nn_val, nn_vals[0], nn_vals[-1])
        AA_1d = np.array([np.interp(nn_val_clipped, nn_vals, AA_grid[i, :])
                          for i in range(n_eta)])
        Bf_1d = np.array([np.interp(nn_val_clipped, nn_vals, Bf_grid[i, :])
                          for i in range(n_eta)])
        Br_1d = np.array([np.interp(nn_val_clipped, nn_vals, Br_grid[i, :])
                          for i in range(n_eta)])

        self._eta_grid = jnp.array(eta_vals)
        self._AA_1d = jnp.array(AA_1d)
        self._Bf_1d = jnp.array(Bf_1d)
        self._Br_1d = jnp.array(Br_1d)

    def compute_log10_lbol_rphot(self, x, t_days):
        nn = float(self.nn)
        delta = float(self.delta)
        eff = self.efficiency

        log10_mej = x["log10_mej"] + _LOG10_MSUN    # grams
        log10_csm = x["log10_csm_mass"] + _LOG10_MSUN  # grams
        log10_vej = x["log10_vej"] + _LOG10_KM_CGS  # cm/s
        eta = x["eta"]
        log10_rho = x["log10_rho"]
        log10_kappa = x["log10_kappa"]
        log10_r0 = x["log10_r0"] + _LOG10_AU_CGS    # cm

        # Chevalier coefficients via 1D interp on eta
        AA = jnp.interp(eta, self._eta_grid, self._AA_1d)
        Bf = jnp.interp(eta, self._eta_grid, self._Bf_1d)
        Br = jnp.interp(eta, self._eta_grid, self._Br_1d)
        log10_AA = jnp.log10(jnp.maximum(AA, 1e-30))
        log10_Bf = jnp.log10(jnp.maximum(Bf, 1e-30))
        log10_Br = jnp.log10(jnp.maximum(Br, 1e-30))

        # --- Geometry in log10 space to avoid float32 overflow ---
        log10_qq = log10_rho + eta * log10_r0
        kappa = jnp.power(10.0, log10_kappa)

        # radius_csm = ((3-eta)/(4*pi*qq)*csm + r0^(3-eta))^(1/(3-eta))
        # Compute each term in log10, then combine
        log10_term1 = (jnp.log10(3.0 - eta) - _LOG10_4PI
                       - log10_qq + log10_csm)
        log10_term2 = (3.0 - eta) * log10_r0
        log10_max_rc = jnp.maximum(log10_term1, log10_term2)
        log10_rc_inner = log10_max_rc + jnp.log10(
            jnp.power(10.0, log10_term1 - log10_max_rc)
            + jnp.power(10.0, log10_term2 - log10_max_rc))
        log10_radius_csm = log10_rc_inner / (3.0 - eta)

        # r_photosphere = |(-2(1-eta)/(3*kappa*qq) + R_csm^(1-eta))^(1/(1-eta))|
        # term_a = -2(1-eta)/(3*kappa*qq) [negative]
        # term_b = R_csm^(1-eta) [positive]
        log10_abs_a = (jnp.log10(2.0 * jnp.abs(1.0 - eta) / 3.0)
                       - log10_kappa - log10_qq)
        log10_b = (1.0 - eta) * log10_radius_csm
        # r_ph_inner = term_b - |term_a| (since term_a is negative)
        # Use log-sub: log10(b-a) = log10(b) + log10(1 - a/b)
        ratio = jnp.power(10.0, log10_abs_a - log10_b)
        ratio = jnp.minimum(ratio, 0.999)  # ensure positive result
        log10_rph_inner = log10_b + jnp.log10(1.0 - ratio)
        log10_r_ph = log10_rph_inner / (1.0 - eta)

        r_photosphere = jnp.power(10.0, log10_r_ph)

        # Optically thick CSM mass:
        # mcst = |4*pi*qq/(3-eta) * (r_ph^(3-eta) - r0^(3-eta))|
        log10_mcst_coeff = _LOG10_4PI + log10_qq - jnp.log10(3.0 - eta)
        log10_rph_term = (3.0 - eta) * log10_r_ph
        log10_r0_term = (3.0 - eta) * log10_r0
        # r_ph^(3-eta) > r0^(3-eta) typically
        log10_diff = log10_rph_term + jnp.log10(
            1.0 - jnp.power(10.0, log10_r0_term - log10_rph_term))
        log10_mcst = log10_mcst_coeff + log10_diff

        # --- Overflow-prone quantities in log10 space ---
        # Esn = 0.3 * vej^2 * mej
        log10_Esn = jnp.log10(0.3) + 2.0 * log10_vej + log10_mej

        # g_n = 1/(4*pi*(nn-delta)) * (2*(5-delta)*(nn-5)*Esn)^((nn-3)/2)
        #       / ((3-delta)*(nn-3)*mej)^((nn-5)/2)
        log10_g_n = (-_LOG10_4PI - jnp.log10(nn - delta)
                     + 0.5 * (nn - 3.0) * (jnp.log10(
                         2.0 * (5.0 - delta) * (nn - 5.0)) + log10_Esn)
                     - 0.5 * (nn - 5.0) * (jnp.log10(
                         (3.0 - delta) * (nn - 3.0)) + log10_mej))

        # --- Shock breakout times in log10(seconds) ---
        ab = nn - eta
        # t_FS
        p_FS = ab / ((nn - 3.0) * (3.0 - eta))
        log10_t_FS_inner = (jnp.log10(3.0 - eta)
                            + (3.0 - nn) / ab * log10_qq
                            + (eta - 3.0) / ab * (log10_AA + log10_g_n)
                            - _LOG10_4PI - (3.0 - eta) * log10_Bf)
        log10_t_FS = p_FS * log10_t_FS_inner + p_FS * log10_mcst

        # t_RS (approximation: ignore the small correction term for nn>>3)
        # t_RS ~ (vej / (Br * (AA*g_n/qq)^(1/ab)))^(ab/(eta-3))
        log10_inner_RS = (log10_vej - log10_Br
                          - (1.0 / ab) * (log10_AA + log10_g_n - log10_qq))
        log10_t_RS = ab / (eta - 3.0) * log10_inner_RS

        t_FS_sec = jnp.power(10.0, log10_t_FS)
        t_RS_sec = jnp.power(10.0, log10_t_RS)

        # --- Shock luminosities in log10 space ---
        t_start = jnp.maximum(t_days[0] * 0.1, 0.01)
        t_end = t_days[-1] * 1.1
        t_int = jnp.linspace(t_start, t_end, self._n_internal)
        t_int_sec = t_int * days_to_seconds + 1.0  # regularization

        alpha_L = (2.0 * nn + 6.0 * eta - nn * eta - 15.0) / ab
        log10_t_sec = jnp.log10(t_int_sec)

        # log10(lbol_FS_coeff) — time-independent part
        log10_FS_coeff = (jnp.log10(2.0 * jnp.pi) - 3.0 * jnp.log10(ab)
                          + (5.0 - eta) / ab * log10_g_n
                          + (nn - 5.0) / ab * log10_qq
                          + 2.0 * jnp.log10(nn - 3.0)
                          + jnp.log10(nn - 5.0)
                          + (5.0 - eta) * log10_Bf
                          + (5.0 - eta) / ab * log10_AA)

        # log10(lbol_RS_coeff)
        log10_RS_coeff = (jnp.log10(2.0 * jnp.pi)
                          + (5.0 - nn) / ab
                            * (log10_AA + log10_g_n - log10_qq)
                          + (5.0 - nn) * log10_Br
                          + log10_g_n
                          + 3.0 * jnp.log10((3.0 - eta) / ab))

        log10_lbol_FS = log10_FS_coeff + alpha_L * log10_t_sec
        log10_lbol_RS = log10_RS_coeff + alpha_L * log10_t_sec

        # Mask by breakout times (set to -30 when inactive)
        mask_FS = t_int_sec < t_FS_sec
        mask_RS = t_int_sec < t_RS_sec
        log10_lbol_FS = jnp.where(mask_FS, log10_lbol_FS, -30.0)
        log10_lbol_RS = jnp.where(mask_RS, log10_lbol_RS, -30.0)

        # Combine: lbol = eff * (lbol_FS + lbol_RS) via log-sum-exp
        log10_max = jnp.maximum(log10_lbol_FS, log10_lbol_RS)
        lbol_sum = (jnp.power(10.0, log10_lbol_FS - log10_max)
                    + jnp.power(10.0, log10_lbol_RS - log10_max))
        log10_lbol = (jnp.log10(eff) + log10_max
                      + jnp.log10(jnp.maximum(lbol_sum, 1e-30)))

        # --- CSM diffusion kernel ---
        log10_beta_csm = jnp.log10(4.0 * jnp.pi**3 / 9.0)
        log10_t0_csm_sec = (log10_kappa + log10_mcst
                            - log10_beta_csm - _LOG10_CCGS - log10_r_ph)
        t0_csm_days = jnp.maximum(
            jnp.power(10.0, log10_t0_csm_sec - _LOG10_DAYS2SEC), 1e-6)

        dt_int = t_int[1] - t_int[0]
        log10_L_scale = jnp.maximum(jnp.max(log10_lbol), 30.0)
        L_n = jnp.power(10.0, log10_lbol - log10_L_scale)

        def _csm_diff_step(W_n, inputs):
            L_n_i, t_i = inputs
            dW = (L_n_i - W_n / t0_csm_days) * dt_int
            W_new = jnp.maximum(W_n + dW, 0.0)
            L_out = W_new / t0_csm_days
            return W_new, L_out

        W0 = L_n[0] * dt_int
        _, L_diff_n = jax.lax.scan(_csm_diff_step, W0, (L_n, t_int))

        # Interpolate to output times
        L_diff_out = jnp.interp(t_days, t_int, L_diff_n)
        log10_L = jnp.log10(jnp.maximum(L_diff_out, 1e-30)) + log10_L_scale
        log10_L = jnp.maximum(log10_L, 0.0)

        # Photosphere: constant from CSM geometry
        log10_R = jnp.full_like(t_days, log10_r_ph)

        return log10_L, log10_R


# ---------------------------------------------------------------------------
# Phenomenological (flux-shape) models — ported from Boom
# ---------------------------------------------------------------------------

class PhenomenologicalModel(AnalyticalModel):
    """Base class for phenomenological light-curve models.

    Unlike physics-based models that compute L_bol + R_phot and pass through
    a blackbody SED, phenomenological models compute a temporal shape function
    S(t) and convert directly to per-band apparent magnitudes.

    Subclasses must set:
        shape_parameter_names : list[str] — shared temporal shape parameters
        has_baseline : bool — whether the model has a baseline flux component

    and implement ``compute_shape(self, x, t_days) -> Array``.
    """

    shape_parameter_names: list[str]
    has_baseline: bool = False

    def __init__(self, filters: list[str], times: Array = None):
        # Build filter list via parent (sets self.filters, self.Filters)
        super().__init__(filters, times)
        # Build parameter_names dynamically from shape params + per-band params
        self._build_parameter_names()

    def _build_parameter_names(self):
        names = list(self.shape_parameter_names)
        for fname in self.filters:
            names.append(f"amp_mag_{fname}")
            if self.has_baseline:
                names.append(f"base_mag_{fname}")
        self.parameter_names = names

    def add_filter(self, filters):
        super().add_filter(filters)
        self._build_parameter_names()

    def compute_shape(self, x: dict[str, Array],
                      t_days: Array) -> Array:
        """Return the temporal shape function S(t) >= 0."""
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def predict(self, x: dict[str, Array]) -> tuple[Array, dict[str, Array]]:
        t_days = self.times
        shape = self.compute_shape(x, t_days)

        mag_app = {}
        for fname in self.filters:
            amp_mag = x[f"amp_mag_{fname}"]
            if self.has_baseline:
                base_mag = x[f"base_mag_{fname}"]
                # total_flux = 10^(-0.4*amp_mag) * shape + 10^(-0.4*base_mag)
                total_flux = (jnp.power(10.0, -0.4 * amp_mag) * shape
                              + jnp.power(10.0, -0.4 * base_mag))
                mag_app[fname] = -2.5 * jnp.log10(jnp.maximum(total_flux, 1e-30))
            else:
                # mag = amp_mag - 2.5 * log10(max(shape, 1e-30))
                mag_app[fname] = amp_mag - 2.5 * jnp.log10(
                    jnp.maximum(shape, 1e-30))

        return t_days, mag_app


class BazinModel(PhenomenologicalModel):
    """Bazin et al. phenomenological light-curve model.

    Shape: exp(-dt/tau_fall) * sigmoid(dt/tau_rise)

    Parameters (per-band): amp_mag_{filter}, base_mag_{filter}
    Shape parameters: t0, log10_tau_rise, log10_tau_fall
    """

    shape_parameter_names = ["t0", "log10_tau_rise", "log10_tau_fall"]
    has_baseline = True

    def __init__(self, filters, times=None):
        if times is None:
            times = jnp.linspace(0.01, 100.0, 200)
        super().__init__(filters, times)

    def compute_shape(self, x, t_days):
        t0 = x["t0"]
        tau_rise = jnp.power(10.0, x["log10_tau_rise"])
        tau_fall = jnp.power(10.0, x["log10_tau_fall"])
        dt = t_days - t0
        return jnp.exp(-dt / tau_fall) * jax.nn.sigmoid(dt / tau_rise)


class VillarModel(PhenomenologicalModel):
    """Villar et al. phenomenological light-curve model.

    Piecewise shape with smooth sigmoid transition at gamma.

    Shape parameters: t0, log10_tau_rise, log10_tau_fall, beta_slope, log10_gamma
    """

    shape_parameter_names = ["t0", "log10_tau_rise", "log10_tau_fall",
                             "beta_slope", "log10_gamma"]
    has_baseline = False

    def __init__(self, filters, times=None):
        if times is None:
            times = jnp.linspace(0.01, 150.0, 200)
        super().__init__(filters, times)

    def compute_shape(self, x, t_days):
        t0 = x["t0"]
        tau_rise = jnp.power(10.0, x["log10_tau_rise"])
        tau_fall = jnp.power(10.0, x["log10_tau_fall"])
        beta = x["beta_slope"]
        gamma = jnp.power(10.0, x["log10_gamma"])

        phase = t_days - t0
        sig_rise = jax.nn.sigmoid(phase / tau_rise)
        w = jax.nn.sigmoid(10.0 * (phase - gamma))
        piece_left = 1.0 - beta * phase
        piece_right = (1.0 - beta * gamma) * jnp.exp((gamma - phase) / tau_fall)
        return sig_rise * jnp.maximum(
            (1.0 - w) * piece_left + w * piece_right, 1e-30)


class PhenomenologicalTDEModel(PhenomenologicalModel):
    """Phenomenological TDE light-curve model.

    Sigmoid rise with power-law decay.

    Shape parameters: t0, log10_tau_rise, log10_tau_fall, alpha_decay
    """

    shape_parameter_names = ["t0", "log10_tau_rise", "log10_tau_fall",
                             "alpha_decay"]
    has_baseline = True

    def __init__(self, filters, times=None):
        if times is None:
            times = jnp.linspace(0.01, 200.0, 200)
        super().__init__(filters, times)

    def compute_shape(self, x, t_days):
        t0 = x["t0"]
        tau_rise = jnp.power(10.0, x["log10_tau_rise"])
        tau_fall = jnp.power(10.0, x["log10_tau_fall"])
        alpha = x["alpha_decay"]

        phase = t_days - t0
        sig = jax.nn.sigmoid(phase / tau_rise)
        phase_soft = jax.nn.softplus(phase) + 1e-6
        decay = jnp.power(1.0 + phase_soft / tau_fall, -alpha)
        return sig * decay


class AfterglowModel(PhenomenologicalModel):
    """Smooth broken power-law afterglow model.

    Transitions from r^(-alpha_1) at early times to r^(-alpha_2) at late times.

    Shape parameters: t0, log10_t_break, alpha_1, alpha_2
    """

    shape_parameter_names = ["t0", "log10_t_break", "alpha_1", "alpha_2"]
    has_baseline = False

    def __init__(self, filters, times=None):
        if times is None:
            times = jnp.linspace(0.01, 300.0, 200)
        super().__init__(filters, times)

    def compute_shape(self, x, t_days):
        t0 = x["t0"]
        t_break = jnp.power(10.0, x["log10_t_break"])
        alpha_1 = x["alpha_1"]
        alpha_2 = x["alpha_2"]

        phase = t_days - t0
        phase_soft = jax.nn.softplus(phase) + 1e-6
        r = phase_soft / t_break
        ln_r = jnp.log(r)
        u1 = jnp.exp(2.0 * alpha_1 * ln_r)
        u2 = jnp.exp(2.0 * alpha_2 * ln_r)
        return jnp.power(u1 + u2, -0.5)
