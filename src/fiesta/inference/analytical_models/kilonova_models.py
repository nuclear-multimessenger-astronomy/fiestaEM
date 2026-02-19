"""Kilonova analytical light-curve models.

Reference:
    Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/kilonova_models.py
    NMMA: https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/nmma/em/lightcurve_generation.py
"""

import jax
import jax.numpy as jnp

from fiesta.constants import c_cgs, days_to_seconds

from fiesta.inference.analytical_models.base import (
    AnalyticalModel,
    _magnetar_luminosity,
    _LOG10_MSUN, _LOG10_RSUN, _LOG10_CCGS, _LOG10_4PI,
)


class MetzgerModel(AnalyticalModel):
    """Efficient single-zone kilonova model (Metzger 2017).

    Reference:
        Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/kilonova_models.py
        NMMA: https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/nmma/em/lightcurve_generation.py

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


class MetzgerFullModel(AnalyticalModel):
    """Full multi-shell kilonova model (Metzger 2017).

    Reference:
        Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/kilonova_models.py
        NMMA: https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/nmma/em/lightcurve_generation.py

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

    _n_shells = 100
    _n_internal = 200

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


class OneComponentKilonovaModel(AnalyticalModel):
    """Single-component kilonova with diffusion-integral heating.

    Reference:
        Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/kilonova_models.py

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


class MagnetarBoostedKilonovaModel(AnalyticalModel):
    """Single-zone kilonova with magnetar spin-down heating injection.

    Reference:
        Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/magnetar_driven_ejecta_models.py

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
