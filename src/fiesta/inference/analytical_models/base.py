"""Base classes, constants, and shared helpers for analytical light-curve models.

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


def _gauss_legendre_nodes_weights(n=32):
    """Return n-point Gauss-Legendre nodes/weights on [0, 1]."""
    import numpy as np
    nodes, weights = np.polynomial.legendre.leggauss(n)
    nodes = (nodes + 1.0) / 2.0
    weights = weights / 2.0
    return jnp.array(nodes), jnp.array(weights)


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
                                 jnp.log10(nu_max * 1.05), 100)

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
