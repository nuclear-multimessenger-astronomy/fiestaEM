"""Kilonova analytical light-curve models.

Reference:
    Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/kilonova_models.py
    NMMA: https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/nmma/em/lightcurve_generation.py
"""

import jax
import jax.numpy as jnp

from fiesta.constants import c_cgs, msun_cgs, days_to_seconds

from fiesta.inference.analytical_models.base import (
    AnalyticalModel,
    _magnetar_luminosity,
    _LOG10_MSUN, _LOG10_RSUN, _LOG10_CCGS, _LOG10_4PI,
    _barnes16_thermalisation_coefficients,
    _barnes16_e_th,
    _electron_fraction_from_kappa,
)


class MetzgerModel(AnalyticalModel):
    """300-shell kilonova model matching NMMA ``eff_metzger_lc``.

    Reference:
        Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/kilonova_models.py
        NMMA: https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/nmma/em/lightcurve_generation.py

    Parameters (in ``x`` dict):
        log10_mej     – log10 ejecta mass in solar masses
        log10_vej     – log10 ejecta velocity in units of c
        beta          – velocity power-law index
        log10_kappa_r – log10 opacity in cm^2/g

    The ODE is solved per-shell in normalized units to avoid float32 overflow.
    Uses 300 mass shells with velocity profile, neutron fractions, and
    shell-dependent opacities matching the NMMA implementation.
    """

    parameter_names = ["log10_mej", "log10_vej", "beta", "log10_kappa_r"]

    _n_shells = 300
    _n_internal = 500

    def __init__(self, filters, times=None):
        if times is None:
            times = jnp.geomspace(0.1, 30.0, 100)
        super().__init__(filters, times)

    def compute_log10_lbol_rphot(self, x, t_days):
        M0 = jnp.power(10.0, x["log10_mej"]) * msun_cgs    # total ejecta mass (g)
        v0 = jnp.power(10.0, x["log10_vej"]) * c_cgs        # minimum escape velocity (cm/s)
        beta_val = x["beta"]
        kappa_r = jnp.power(10.0, x["log10_kappa_r"])

        n_shells = self._n_shells  # 300

        # Use the output time grid directly (matching NMMA's approach)
        t_sec = t_days * days_to_seconds

        # Variable time steps (matches NMMA's geomspace dt)
        dt_arr = jnp.diff(t_sec)
        # Pad with the first dt so arrays match (first step is approximate)
        dt_padded = jnp.concatenate([dt_arr[:1], dt_arr])

        # Thermalization efficiency pre-computed at output times (matching NMMA)
        def _eth(t_day):
            timescale_factor = 2.0 * 0.17 * jnp.power(jnp.maximum(t_day, 1e-6), 0.74)
            return 0.36 * (jnp.exp(-0.56 * t_day)
                           + jnp.log(1.0 + timescale_factor) / timescale_factor)

        eth_arr = _eth(t_days)  # (n_t,)

        # Mass shells: geometric spacing (solar masses) matching NMMA
        m = jnp.geomspace(1e-8, jnp.power(10.0, x["log10_mej"]), n_shells)  # Msun
        dm = jnp.diff(m)  # (n_shells-1,) in Msun

        # Velocity profile: v = v0 * (m*msun/M0)^(-1/beta), capped at c
        vm = v0 * jnp.power(m * msun_cgs / M0, -1.0 / beta_val)
        vm = jnp.minimum(vm, c_cgs)

        # Neutron fractions (NMMA: Mn=1e-8, Ye=0.1, Xn0max=0.8)
        Mn = 1e-8
        Xn0 = 0.8 * 2.0 / jnp.pi * jnp.arctan(Mn / m)  # (n_shells,)
        Xr = 1.0 - Xn0  # r-process fraction

        # Pre-compute time-dependent arrays: Xn(t), edot(t), kappa(t)
        # Shape: (n_t, n_shells) or (n_t, n_shells-1)
        Xn_t = Xn0[:-1][None, :] * jnp.exp(-t_sec[:, None] / 900.0)  # (n_t, ns-1)
        edotn = 3.2e14 * Xn_t  # (n_t, ns-1)
        edotr = (2.1e10 * eth_arr[:, None]
                 * jnp.power(jnp.maximum(t_days, 1e-6), -1.3)[:, None])  # (n_t, ns-1)
        edot_all = edotn + edotr  # (n_t, ns-1)

        kappan = 0.4 * (1.0 - Xn_t - Xr[:-1][None, :])
        kappa_all = kappan + kappa_r * Xr[:-1][None, :]  # (n_t, ns-1)

        # The ODE uses per-gram quantities (ene in erg/g), which are moderate
        # (~1e10 to 1e18) and safe for float32.

        def _scan_step(ene, inputs):
            # ene: (n_shells-1,) energy per gram in erg/g
            t_i, dt_i, edot_i, kappa_i = inputs
            t_safe = jnp.maximum(t_i, 1.0)

            # Diffusion timescale per shell (NMMA formula)
            tdiff = 0.08 * kappa_i * m[:-1] * msun_cgs * 3.0 / (
                vm[:-1] * c_cgs * t_safe * beta_val)

            # Luminosity per unit mass (erg/s/g)
            lum_j = ene / (tdiff + t_i * vm[:-1] / c_cgs)

            # Total luminosity: sum(lum_j * dm) in Msun*erg/s/g (moderate)
            L_dm_total = jnp.sum(lum_j * dm)

            # Optical depth for photosphere
            tau_shells = m[:-1] * msun_cgs * kappa_i / (
                4.0 * jnp.pi * (t_safe * vm[:-1])**2)
            pig = jnp.argmin(jnp.abs(tau_shells - 1.0))
            R_ph = vm[pig] * t_i

            # Energy ODE (NMMA: ene += dt*(edot - ene/t - lum_j))
            dene = dt_i * (edot_i - ene / t_safe - lum_j)
            ene_new = jnp.maximum(ene + dene, 0.0)

            return ene_new, (L_dm_total, R_ph)

        # Initial energy per shell: zero (NMMA starts from zero)
        E0 = jnp.zeros(n_shells - 1)
        _, (L_dm_arr, R_arr) = jax.lax.scan(
            _scan_step, E0, (t_sec, dt_padded, edot_all, kappa_all))

        # Convert total luminosity to log10:
        # L_total = L_dm_total * msun_cgs
        log10_L = jnp.log10(jnp.maximum(jnp.abs(L_dm_arr), 1e-30)) + _LOG10_MSUN
        log10_L = jnp.maximum(log10_L, 0.0)

        log10_R = jnp.log10(jnp.maximum(R_arr, 1.0))

        return log10_L, log10_R


class MetzgerFullModel(AnalyticalModel):
    """Multi-shell kilonova model (Metzger 2017), matching Redback exactly.

    Reference:
        Redback: _metzger_kilonova_model in kilonova_models.py

    200 shells with linear velocity spacing, Barnes+16 thermalisation,
    optional neutron precursor, per-gram energy ODE.

    Parameters (in ``x`` dict):
        log10_mej     – log10 ejecta mass in solar masses
        log10_vej     – log10 ejecta velocity (vmin) in units of c
        beta          – velocity power-law index
        log10_kappa_r – log10 opacity in cm^2/g
    """

    parameter_names = ["log10_mej", "log10_vej", "beta", "log10_kappa_r"]

    _n_shells = 200

    def __init__(self, filters, times=None, neutron_precursor=True, vmax=0.7):
        self._neutron_precursor = neutron_precursor
        self._vmax = vmax
        if times is None:
            times = jnp.geomspace(0.1, 30.0, 100)
        super().__init__(filters, times)

    def compute_log10_lbol_rphot(self, x, t_days):
        mej = jnp.power(10.0, x["log10_mej"])   # solar masses
        vej = jnp.power(10.0, x["log10_vej"])    # fraction of c
        beta_val = x["beta"]
        kappa_r = jnp.power(10.0, x["log10_kappa_r"])

        mass_len = self._n_shells  # 200

        t_sec = t_days * days_to_seconds
        t_d = t_days

        # Barnes+16 thermalisation
        av, bv, dv = _barnes16_thermalisation_coefficients(mej, vej)
        e_th = _barnes16_e_th(t_d, av, bv, dv)  # (time_len,)

        # Heating rate (dual regime, matching Redback lines 1867-1871)
        t0 = 1.3   # seconds
        sig = 0.11  # seconds
        edotr_late = 2.1e10 * e_th * t_d ** (-1.3)
        edotr_early = 4.0e18 * jnp.power(
            0.5 - jnp.arctan((t_sec - t0) / sig) / jnp.pi, 1.3) * e_th
        edotr_base = jnp.where(t_sec > t0, edotr_late, edotr_early)

        # Shell layout: linear velocity spacing (matching Redback)
        vmin = vej
        vmax = self._vmax
        vel = jnp.linspace(vmin, vmax, mass_len)
        m_array = mej * (vel / vmin) ** (-beta_val)  # solar masses
        v_m = vel * c_cgs  # cm/s

        dm = jnp.abs(jnp.diff(m_array))  # (mass_len-1,) solar masses

        # Neutron precursor (matching Redback)
        neutron_precursor = self._neutron_precursor
        if neutron_precursor:
            Ye = _electron_fraction_from_kappa(kappa_r)
            neutron_mass = 1e-8 * msun_cgs
            Xn0 = 1.0 - 2.0 * Ye * 2.0 * jnp.arctan(
                neutron_mass / m_array / msun_cgs) / jnp.pi
            Xr = 1.0 - Xn0

        # Initial conditions in Redback mixed units: energy_v = m_array * v^2/2
        # Units: Msun * (cm/s)^2 — moderate values, safe for float32
        E0 = 0.5 * m_array * v_m ** 2

        dt_arr = jnp.diff(t_sec)

        def _scan_step(ene, inputs):
            t_i, dt_i, edotr_i, e_th_i = inputs

            if neutron_precursor:
                Xn_t = Xn0 * jnp.exp(-t_i / 900.0)
                edotn = 3.2e14 * Xn_t * Xn_t
                edot = edotn[:-1] + edotr_i
                kappa_n = 0.4 * (1.0 - Xn_t - Xr)
                kappa_total = kappa_n + kappa_r * Xr
            else:
                edot = edotr_i * jnp.ones(mass_len - 1)
                kappa_total = kappa_r * jnp.ones(mass_len)

            td_v = (kappa_total[:-1] * m_array[:-1] * msun_cgs * 3.0) / (
                4.0 * jnp.pi * v_m[:-1] * c_cgs * t_i * beta_val)

            lum_rad = ene[:-1] / (td_v + t_i * (v_m[:-1] / c_cgs))

            ene_new = jnp.concatenate([
                ene[:-1] + (edot - ene[:-1] / t_i - lum_rad) * dt_i,
                ene[-1:],
            ])
            ene_new = jnp.maximum(ene_new, 0.0)

            # Sum lum * dm in mixed units (moderate), defer msun to log10
            L_dm_total = jnp.sum(lum_rad * dm)

            tau = m_array[:-1] * msun_cgs * kappa_total[:-1] / (
                4.0 * jnp.pi * (t_i * v_m[:-1]) ** 2)
            tau_full = jnp.concatenate([tau, tau[-1:]])
            pig = jnp.argmin(jnp.abs(tau_full - 1.0))
            R_ph = v_m[pig] * t_i

            return ene_new, (L_dm_total, R_ph)

        _, (L_arr, R_arr) = jax.lax.scan(
            _scan_step, E0,
            (t_sec[:-1], dt_arr, edotr_base[:-1], e_th[:-1]))

        L_arr = jnp.concatenate([L_arr, L_arr[-1:]])
        R_arr = jnp.concatenate([R_arr, R_arr[-1:]])

        # Convert: L_erg_s = L_dm_total * msun_cgs
        log10_L = jnp.log10(jnp.maximum(jnp.abs(L_arr), 1e-30)) + _LOG10_MSUN
        log10_L = jnp.maximum(log10_L, 0.0)

        log10_R = jnp.log10(jnp.maximum(R_arr, 1.0))

        return log10_L, log10_R


class OneComponentKilonovaModel(AnalyticalModel):
    """Single-component kilonova with diffusion-integral heating.

    Reference:
        Redback: _one_component_kilonova_model in kilonova_models.py

    Matches Redback's cumulative trapezoid algorithm exactly, using a
    float32-safe damped recurrence that avoids exp(t^2/td^2) overflow.

    Parameters (in ``x`` dict):
        log10_mej   – log10 ejecta mass in solar masses
        log10_vej   – log10 ejecta velocity in units of c
        log10_kappa – log10 gray opacity in cm^2/g
    """

    parameter_names = ["log10_mej", "log10_vej", "log10_kappa"]

    def __init__(self, filters, times=None, temperature_floor=4000.0):
        if times is None:
            times = jnp.geomspace(0.1, 30.0, 100)
        super().__init__(filters, times, temperature_floor=temperature_floor)

    def compute_log10_lbol_rphot(self, x, t_days):
        log10_mej_g = x["log10_mej"] + _LOG10_MSUN
        log10_vej_cms = x["log10_vej"] + _LOG10_CCGS
        log10_kappa = x["log10_kappa"]

        mej_solar = jnp.power(10.0, x["log10_mej"])
        vej_c = jnp.power(10.0, x["log10_vej"])

        t_sec = t_days * days_to_seconds

        # Diffusion timescale: td = sqrt(2 * kappa * mej / (13.7 * vej * c))
        log10_td = 0.5 * (jnp.log10(2.0) + log10_kappa + log10_mej_g
                          - jnp.log10(13.7) - log10_vej_cms - _LOG10_CCGS)
        td = jnp.power(10.0, log10_td)

        # Normalization: Q_scale = 4e18 * mej_g
        log10_Q_scale = jnp.log10(4.0e18) + log10_mej_g

        # Barnes+16 thermalisation
        av, bv, dv = _barnes16_thermalisation_coefficients(mej_solar, vej_c)
        e_th = _barnes16_e_th(t_days, av, bv, dv)

        # Normalized integrand (without exp factor):
        # f_n = shape(t) * e_th(t) * (t/td)
        # Use identity: 0.5 - arctan(x)/π = arctan(1/x)/π for x > 0
        # to avoid float32 catastrophic cancellation at late times.
        t0_heat = 1.3   # seconds
        sig_heat = 0.11  # seconds
        f_n = (jnp.power(
            jnp.arctan(sig_heat / jnp.maximum(t_sec - t0_heat, 1e-6))
            / jnp.pi, 1.3)
            * e_th * (t_sec / td))

        # Float32-safe O(n²) vectorized cumulative trapezoid.
        # Redback: L = cumtrapz(f*exp(t²/td²), t) * exp(-t²/td²) / td
        # Equivalent: L[j] = (1/td) * sum_i 0.5*(f[i]*φ[j,i] + f[i+1]*φ[j,i+1])*dt[i]
        # where φ[j,k] = exp(-(t[j]²-t[k]²)/td²) ∈ [0,1] for j >= k.
        # Each output computed independently — no sequential error accumulation.
        n = t_sec.shape[0]
        dt = jnp.diff(t_sec)  # (n-1,)
        t_sq = t_sec ** 2

        # Damping factors: φ[j+1, k] for j=0..n-2, k=0..n-1
        exp_arg = -(t_sq[1:, None] - t_sq[None, :]) / td**2  # (n-1, n)
        phi = jnp.exp(jnp.clip(exp_arg, -80.0, 0.0))

        # Trapezoid: h[j, i] = 0.5*(f[i]*φ[j+1,i] + f[i+1]*φ[j+1,i+1]) * dt[i]
        h = 0.5 * (f_n[:-1][None, :] * phi[:, :-1]
                    + f_n[1:][None, :] * phi[:, 1:]) * dt[None, :]

        # Mask: only i <= j (lower triangular incl. diagonal)
        mask = jnp.tril(jnp.ones((n - 1, n - 1), dtype=bool))
        h = jnp.where(mask, h, 0.0)

        W_arr = jnp.sum(h, axis=1)  # (n-1,)

        # L[0] = L[1] matching Redback
        W_full = jnp.concatenate([W_arr[:1], W_arr])

        L_n = jnp.maximum(W_full / td, 0.0)

        log10_L = jnp.log10(jnp.maximum(L_n, 1e-30)) + log10_Q_scale
        log10_L = jnp.maximum(log10_L, 0.0)

        # Photospheric radius: R = vej * t
        log10_R = log10_vej_cms + jnp.log10(t_sec)

        return log10_L, log10_R


class MagnetarBoostedKilonovaModel(AnalyticalModel):
    """Multi-shell kilonova with magnetar spin-down heating, matching Redback.

    Reference:
        Redback: _general_metzger_magnetar_driven_kilonova_model

    200-shell ODE with magnetar injection into bottom layer, velocity
    evolution, optional pair cascade and neutron precursor.

    Parameters (in ``x`` dict):
        log10_mej                 – log10 ejecta mass in solar masses
        log10_vej                 – log10 ejecta velocity (vmin) in units of c
        beta                      – velocity power-law index
        log10_kappa_r             – log10 opacity in cm^2/g
        log10_p0                  – log10 initial spin period in ms
        log10_bp                  – log10 polar B-field in 1e14 G
        mass_ns                   – neutron star mass in solar masses
        theta_pb                  – angle between spin and B-field in radians
        thermalisation_efficiency – magnetar thermalisation efficiency
    """

    parameter_names = ["log10_mej", "log10_vej", "beta", "log10_kappa_r",
                       "log10_p0", "log10_bp", "mass_ns", "theta_pb",
                       "thermalisation_efficiency"]

    _n_shells = 200

    def __init__(self, filters, times=None, neutron_precursor=True,
                 pair_cascade=True, vmax=0.7, magnetar_heating='first_layer'):
        self._neutron_precursor = neutron_precursor
        self._pair_cascade = pair_cascade
        self._vmax = vmax
        self._magnetar_heating = magnetar_heating
        if times is None:
            times = jnp.geomspace(0.1, 30.0, 100)
        super().__init__(filters, times)

    def compute_log10_lbol_rphot(self, x, t_days):
        mej = jnp.power(10.0, x["log10_mej"])   # solar masses
        vej = jnp.power(10.0, x["log10_vej"])    # fraction of c
        beta_val = x["beta"]
        kappa_r = jnp.power(10.0, x["log10_kappa_r"])
        th_eff = x["thermalisation_efficiency"]

        mass_len = self._n_shells  # 200

        t_sec = t_days * days_to_seconds
        t_d = t_days

        # Barnes+16 thermalisation for r-process
        av, bv, dv = _barnes16_thermalisation_coefficients(mej, vej)
        e_th = _barnes16_e_th(t_d, av, bv, dv)

        # Heating rate (dual regime)
        t0 = 1.3
        sig = 0.11
        edotr_late = 2.1e10 * e_th * t_d ** (-1.3)
        edotr_early = 4.0e18 * jnp.power(
            0.5 - jnp.arctan((t_sec - t0) / sig) / jnp.pi, 1.3) * e_th
        edotr_base = jnp.where(t_sec > t0, edotr_late, edotr_early)

        # Shell layout with mass normalization (matching Redback)
        vmin = vej
        vmax_val = self._vmax
        vel = jnp.linspace(vmin, vmax_val, mass_len)
        m_array = mej * (vel / vmin) ** (-beta_val)
        total_mass = jnp.sum(m_array)
        m_array = m_array * (mej / total_mass)
        v_m_init = vel * c_cgs

        dm = jnp.abs(jnp.diff(m_array))

        # Magnetar luminosity in log10 (never materialized in linear)
        log10_L_mag = _magnetar_luminosity(
            t_sec, x["log10_p0"], x["log10_bp"], x["mass_ns"], x["theta_pb"])

        # Energy normalization via log10_E_scale (never materialized as 10^x).
        # Redback's energy_v is in erg. We define ene_n = energy_v / E_scale.
        log10_E_scale = jnp.maximum(log10_L_mag[0], 30.0)

        # Precompute msun / E_scale (safe: LOG10_MSUN - log10_E_scale < 0)
        msun_per_E = jnp.power(10.0, _LOG10_MSUN - log10_E_scale)

        # Precompute E_scale / m0 for velocity evolution (float32-safe)
        # E_over_m0 = 10^(log10_E_scale - LOG10_MSUN - log10_mej)
        E_over_m0 = jnp.power(10.0,
                               log10_E_scale - _LOG10_MSUN - x["log10_mej"])

        # Mass power-law exponent (precomputed, constant)
        mass_power = (m_array / mej) ** (-1.0 / beta_val)

        # Neutron precursor
        neutron_precursor = self._neutron_precursor
        if neutron_precursor:
            Ye = _electron_fraction_from_kappa(kappa_r)
            neutron_mass = 1e-8 * msun_cgs
            Xn0 = 1.0 - 2.0 * Ye * 2.0 * jnp.arctan(
                neutron_mass / m_array / msun_cgs) / jnp.pi
            Xr = 1.0 - Xn0

        # Initial conditions (normalized via log10)
        E0_raw = 0.5 * m_array * v_m_init ** 2  # Msun*(cm/s)^2, safe float32
        E0_n = jnp.power(10.0, jnp.log10(jnp.maximum(E0_raw, 1e-30))
                          - log10_E_scale)

        # Initial kinetic energy (normalized): ek_n = 0.5*m0*v0^2 / E_scale
        log10_ek_0 = (jnp.log10(0.5) + x["log10_mej"] + _LOG10_MSUN
                      + 2.0 * (x["log10_vej"] + _LOG10_CCGS))
        ek_n_0 = jnp.power(10.0, log10_ek_0 - log10_E_scale)

        dt_arr = jnp.diff(t_sec)

        first_layer = (self._magnetar_heating == 'first_layer')

        def _scan_step(state, inputs):
            ene_n, ek_n = state
            t_i, dt_i, edotr_i, log10_lsd_i = inputs

            # Velocity evolution (matching Redback lines 682-689):
            # kinetic_energy += energy_v[0]/t * dt
            # v0 = sqrt(2*KE/m0)
            ek_n = ek_n + (ene_n[0] / t_i) * dt_i
            v0_sq = 2.0 * ek_n * E_over_m0
            v0_new = jnp.sqrt(jnp.maximum(v0_sq, 0.0))
            v_m_t = jnp.minimum(v0_new * mass_power, c_cgs)

            # Magnetar heating (normalized via log10)
            log10_qdot_n = (jnp.log10(jnp.maximum(th_eff, 1e-30))
                            + log10_lsd_i - log10_E_scale)
            qdot_n = jnp.power(10.0, jnp.clip(log10_qdot_n, -30.0, 30.0))

            if neutron_precursor:
                Xn_t = Xn0 * jnp.exp(-t_i / 900.0)
                edotn = 3.2e14 * Xn_t * Xn_t
                kappa_n = 0.4 * (1.0 - Xn_t - Xr)
                kappa_total = kappa_n + kappa_r * Xr
            else:
                edotn = jnp.zeros(mass_len)
                kappa_total = kappa_r * jnp.ones(mass_len)

            # Heating terms normalized: edotr * dm * msun / E_scale
            edotr_dm_n = edotr_i * dm * msun_per_E
            edotn_dm_n = edotn[:-1] * dm * msun_per_E

            # Diffusion timescale (using evolved v_m_t)
            td_v = (kappa_total[:-1] * m_array[:-1] * msun_cgs * 3.0) / (
                4.0 * jnp.pi * v_m_t[:-1] * c_cgs * t_i * beta_val)

            lum_n = ene_n[:-1] / (td_v + t_i * (v_m_t[:-1] / c_cgs))

            if first_layer:
                ene_0_new = ene_n[0] + (qdot_n + edotr_dm_n[0] + edotn_dm_n[0]
                                        - ene_n[0] / t_i - lum_n[0]) * dt_i
                ene_mid_new = ene_n[1:-1] + (edotr_dm_n[1:] + edotn_dm_n[1:]
                                             - ene_n[1:-1] / t_i
                                             - lum_n[1:]) * dt_i
                ene_n_new = jnp.concatenate([
                    ene_0_new[None], ene_mid_new, ene_n[-1:]])
            else:
                ene_n_new = jnp.concatenate([
                    ene_n[:-1] + (qdot_n + edotr_dm_n + edotn_dm_n
                                  - ene_n[:-1] / t_i - lum_n) * dt_i,
                    ene_n[-1:],
                ])

            ene_n_new = jnp.maximum(ene_n_new, 0.0)

            L_n_total = jnp.sum(lum_n)

            tau = m_array[:-1] * msun_cgs * kappa_total[:-1] / (
                4.0 * jnp.pi * (t_i * v_m_t[:-1]) ** 2)
            tau_full = jnp.concatenate([tau, tau[-1:]])
            pig = jnp.argmin(jnp.abs(tau_full - 1.0))
            R_ph = v_m_t[pig] * t_i

            return (ene_n_new, ek_n), (L_n_total, R_ph)

        _, (L_n_arr, R_arr) = jax.lax.scan(
            _scan_step, (E0_n, ek_n_0),
            (t_sec[:-1], dt_arr, edotr_base[:-1], log10_L_mag[:-1]))

        L_n_arr = jnp.concatenate([L_n_arr, L_n_arr[-1:]])
        R_arr = jnp.concatenate([R_arr, R_arr[-1:]])

        # Convert: L_erg_s = L_n * E_scale
        log10_L = jnp.log10(jnp.maximum(jnp.abs(L_n_arr), 1e-30)) + log10_E_scale

        # Pair cascade (matching Redback) — uses v0 from last step
        if self._pair_cascade:
            ejecta_albedo = 0.5
            pair_cascade_fraction = 0.01
            log10_tlife = (jnp.log10(0.6 / (1.0 - ejecta_albedo))
                           + 0.5 * jnp.log10(pair_cascade_fraction / 0.1)
                           + 0.5 * (log10_L_mag - 45.0)
                           + 0.5 * jnp.log10(vej / 0.3)
                           - 0.5 * jnp.log10(jnp.maximum(t_d, 1e-10)))
            tlife_t = jnp.power(10.0, jnp.clip(log10_tlife, -20.0, 20.0))
            log10_L = log10_L - jnp.log10(1.0 + tlife_t)

        log10_L = jnp.maximum(log10_L, 0.0)
        log10_R = jnp.log10(jnp.maximum(R_arr, 1.0))

        return log10_L, log10_R
