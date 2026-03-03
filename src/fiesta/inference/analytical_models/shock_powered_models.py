"""Shock-powered analytical light-curve models.

Reference:
    Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/shock_powered_models.py
    NMMA: https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/nmma/em/lightcurve_generation.py
"""

import jax
import jax.numpy as jnp

from fiesta.constants import c_cgs, msun_cgs, Rsun_cgs, days_to_seconds

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
        """Full Piro (2021) shock cooling with n=10, delta=1.1.

        Matches NMMA ``sc_bol_lc`` exactly. All overflow-prone quantities
        are computed in log10 space to stay within float32 range.
        """
        n = 10.0
        delta = 1.1
        kappa = self._kappa  # 0.2 cm^2/g
        log10_kappa = jnp.log10(kappa)

        # Physical parameters in log10-CGS
        log10_Me = x["log10_Menv"] + _LOG10_MSUN     # log10(grams)
        log10_Re = x["log10_Renv"] + _LOG10_RSUN     # log10(cm)
        log10_Ee = x["log10_Ee"]                      # log10(erg)

        # Piro (2021) constants
        K = (n - 3.0) * (3.0 - delta) / (4.0 * jnp.pi * (n - delta))

        # vt = sqrt(((n-5)(5-delta)/((n-3)(3-delta))) * 2*Ee/Me)
        vel_coeff = (n - 5.0) * (5.0 - delta) / ((n - 3.0) * (3.0 - delta))
        log10_vt = 0.5 * (jnp.log10(vel_coeff * 2.0) + log10_Ee - log10_Me)

        # td = sqrt(3*kappa*K*Me / ((n-1)*vt*c))
        log10_td = 0.5 * (jnp.log10(3.0 * kappa * K) + log10_Me
                          - jnp.log10(n - 1.0) - log10_vt - _LOG10_CCGS)
        td = jnp.power(10.0, log10_td)  # moderate: ~1e4-1e5

        # Time in seconds
        t = t_days * days_to_seconds
        t = jnp.maximum(t, 1.0)
        log10_t = jnp.log10(t)

        # prefactor = pi*(n-1)/(3*(n-5)) * c * Re * vt^2 / kappa
        # This can exceed float32 max (~3.4e38), so compute in log10
        log10_prefactor = (jnp.log10(jnp.pi * (n - 1.0) / (3.0 * (n - 5.0)))
                           + _LOG10_CCGS + log10_Re + 2.0 * log10_vt
                           - log10_kappa)

        # L_early = prefactor * (td/t)^(4/(n-2))
        log10_L_early = log10_prefactor + 4.0 / (n - 2.0) * (log10_td - log10_t)

        # L_late = prefactor * exp(-0.5*(t^2/td^2 - 1))
        exp_arg = -0.5 * (t**2 / td**2 - 1.0)
        log10_L_late = log10_prefactor + exp_arg * _LOG10E

        log10_L = jnp.where(t < td, log10_L_early, log10_L_late)
        log10_L = jnp.maximum(log10_L, 0.0)  # floor at 1 erg/s

        # Photospheric radius
        # tph = sqrt(3*kappa*K*Me / (2*(n-1)*vt^2))
        log10_tph = 0.5 * (jnp.log10(3.0 * kappa * K) + log10_Me
                           - jnp.log10(2.0 * (n - 1.0)) - 2.0 * log10_vt)
        tph = jnp.power(10.0, log10_tph)  # moderate: ~1e4

        # R_early = (tph/t)^(2/(n-1)) * vt * t
        log10_R_early = (2.0 / (n - 1.0) * (log10_tph - log10_t)
                         + log10_vt + log10_t)

        # R_late = (1 + (delta-1)/(n-1)*((t/tph)^2-1))^(-1/(delta-1)) * vt*t
        inner = 1.0 + (delta - 1.0) / (n - 1.0) * ((t / tph)**2 - 1.0)
        inner = jnp.maximum(inner, 1e-10)
        log10_R_late = (-1.0 / (delta - 1.0) * jnp.log10(inner)
                        + log10_vt + log10_t)

        log10_R = jnp.where(t < tph, log10_R_early, log10_R_late)

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

        f_sh = jnp.maximum(f_sh, 1e-12)
        cos_theta = jnp.clip(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
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
