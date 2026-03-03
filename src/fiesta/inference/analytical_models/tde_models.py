"""Tidal disruption event (TDE) analytical light-curve models.

Reference:
    Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/tde_models.py
"""

import jax.numpy as jnp

from fiesta.constants import days_to_seconds

from fiesta.inference.analytical_models.base import (
    AnalyticalModel,
    _compute_diffusion_constants,
    _arnett_diffusion_integral,
    _LOG10_MSUN, _LOG10_KM_CGS, _LOG10_DAYS2SEC,
)


class TDEAnalyticalModel(AnalyticalModel):
    """TDE analytical model with t^{-5/3} fallback + Arnett diffusion.

    Reference:
        Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/tde_models.py

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

        # Dense internal time grid in days (log-spaced for better early-time resolution)
        t_start = jnp.maximum(t_days[0] * 0.1, 0.01)
        t_end = t_days[-1] * 1.1
        t_int = jnp.geomspace(t_start, t_end, self._n_internal)

        # Engine: L(t) = l0 / (max(t, t0_turn) * 86400)^{5/3}
        t_eff = jnp.maximum(t_int, t_0_turn)
        log10_t_sec = jnp.log10(t_eff) + _LOG10_DAYS2SEC
        log10_L_engine = log10_l0 - (5.0 / 3.0) * log10_t_sec

        # Diffusion constants
        log10_td, log10_A_trap = _compute_diffusion_constants(
            log10_kappa, log10_kappa_gamma, log10_mej_g, log10_vej_kms)

        # Scale for normalization
        log10_L_scale = jnp.maximum(jnp.max(log10_L_engine), 30.0)

        # Solve diffusion via trapezoidal integral (Redback-matched)
        log10_L_int = _arnett_diffusion_integral(
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
