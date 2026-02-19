"""Shock-powered analytical light-curve models.

Reference:
    Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/shock_powered_models.py
    NMMA: https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/nmma/em/lightcurve_generation.py
"""

import jax.numpy as jnp

from fiesta.constants import c_cgs, days_to_seconds

from fiesta.inference.analytical_models.base import (
    AnalyticalModel,
    _LOG10E, _LOG10_MSUN, _LOG10_RSUN, _LOG10_CCGS, _LOG10_4PI,
    _LOG10_DAYS2SEC,
)


class ShockCoolingModel(AnalyticalModel):
    """Shock-cooling emission following Piro (2021).

    Reference:
        Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/shock_powered_models.py
        NMMA: https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/nmma/em/lightcurve_generation.py

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


class ShockedCocoonModel(AnalyticalModel):
    """Analytical jet cocoon cooling model.

    Reference:
        Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/shock_powered_models.py

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
