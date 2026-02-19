"""Supernova analytical light-curve models.

Reference:
    Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/supernova_models.py
    NMMA: https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/nmma/em/lightcurve_generation.py
"""

import jax
import jax.numpy as jnp

from fiesta.constants import days_to_seconds

from fiesta.inference.analytical_models.base import (
    AnalyticalModel,
    _gauss_legendre_nodes_weights,
    _magnetar_luminosity,
    _compute_diffusion_constants,
    _arnett_diffusion_ode,
    _LOG10E, _LOG10_MSUN, _LOG10_CCGS, _LOG10_4PI, _LOG10_PI,
    _LOG10_KM_CGS, _LOG10_AU_CGS, _LOG10_DAYS2SEC,
)


class ArnettModel(AnalyticalModel):
    """Arnett (1982) Ni56/Co56-powered supernova bolometric model.

    Reference:
        Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/supernova_models.py
        NMMA: https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/nmma/em/lightcurve_generation.py

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

        eps_ni = jnp.power(10.0, self._log10_eps_ni)
        eps_co = jnp.power(10.0, self._log10_eps_co)

        def _log10_lbol_single(t_d):
            x = t_d / tau_m
            z = x * nodes  # GL nodes on [0,1] → z on [0, x]

            # Arnett (1982) integrals with exp(-x^2) absorbed for stability:
            #   exp(-x^2) * A(x) = int_0^x 2z exp(z^2 - x^2 - z*tau_m/tau_ni) dz
            #   exp(-x^2) * B(x) = int_0^x 2z exp(z^2 - x^2 - z*tau_m/tau_co) dz
            # Since z <= x, the exponent z^2 - x^2 <= 0, so bounded in float32.
            A_ni = 2.0 * z * jnp.exp(z**2 - x**2 - z * tau_m / tau_ni)
            I_A = x * jnp.dot(weights, A_ni)

            A_co = 2.0 * z * jnp.exp(z**2 - x**2 - z * tau_m / tau_co)
            I_B = x * jnp.dot(weights, A_co)

            # L = M_ni * [(eps_ni - eps_co)*A + eps_co*B]  (Ni -> Co -> Fe chain)
            total_heating = (eps_ni - eps_co) * I_A + eps_co * I_B

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


class NickelCobaltModel(AnalyticalModel):
    """Ni56/Co56 radioactive decay with Arnett (1982) diffusion.

    Reference:
        Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/supernova_models.py

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


class MagnetarPoweredSNModel(AnalyticalModel):
    """Magnetar spin-down powered supernova with Arnett (1982) diffusion.

    Reference:
        Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/supernova_models.py

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


class CSMInteractionModel(AnalyticalModel):
    """Circumstellar medium interaction model (Chevalier 1982).

    Reference:
        Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/supernova_models.py

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

        table_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  "tables", "csm_table.txt")
        data = np.loadtxt(table_path, delimiter=',')
        # Columns: eta_col(10 unique), nn_col(30 unique), Bf, Br, AA
        # (matches Redback's column order in utils.get_csm_properties)
        eta_col = data[:, 0]
        nn_col = data[:, 1]

        # Reshape: 10 eta values x 30 nn values
        n_eta = len(np.unique(eta_col))
        n_nn = len(np.unique(nn_col))
        Bf_grid = data[:, 2].reshape(n_eta, n_nn)  # (10, 30)
        Br_grid = data[:, 3].reshape(n_eta, n_nn)
        AA_grid = data[:, 4].reshape(n_eta, n_nn)

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

        # t_RS: reverse shock sweep-up time (Chevalier 1982)
        # t_RS = (vej / (Br*(AA*g_n/qq)^(1/ab)) * corr)^(ab/(eta-3))
        # corr = (1 - (3-nn)*mej / (4*pi*vej^(3-nn)*g_n))^(1/(3-nn))
        log10_inner_RS = (log10_vej - log10_Br
                          - (1.0 / ab) * (log10_AA + log10_g_n - log10_qq))
        # Correction factor: for nn > 3, (3-nn) < 0, so the ratio term is
        # negative and the argument is 1 + |ratio| > 1.
        # |ratio| = (nn-3)*mej / (4*pi*vej^(3-nn)*g_n)
        log10_abs_ratio = (jnp.log10(nn - 3.0) + log10_mej
                           - _LOG10_4PI + (nn - 3.0) * log10_vej - log10_g_n)
        abs_ratio = jnp.power(10.0, log10_abs_ratio)
        # arg = 1 + abs_ratio; corr = arg^(1/(3-nn)) = arg^(-1/(nn-3))
        log10_corr = -1.0 / (nn - 3.0) * jnp.log10(1.0 + abs_ratio)
        log10_t_RS = ab / (eta - 3.0) * (log10_inner_RS + log10_corr)

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
