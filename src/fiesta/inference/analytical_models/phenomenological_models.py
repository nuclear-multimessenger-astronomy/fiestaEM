"""Phenomenological (flux-shape) light-curve models.

Reference:
    Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/phenomenological_models.py
    Boom: https://github.com/boom-astro/boom/blob/main/src/fitting/parametric.rs
"""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array

from fiesta.inference.analytical_models.base import (
    AnalyticalModel,
    _LOG10_4PI, _LOG10_SIGMASB,
)


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


class EvolvingBlackbodyModel(AnalyticalModel):
    """Phenomenological model with piecewise power-law T and R evolution.

    Reference:
        Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/phenomenological_models.py

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


class BazinModel(PhenomenologicalModel):
    """Bazin et al. phenomenological light-curve model.

    Reference:
        Boom: https://github.com/boom-astro/boom/blob/main/src/fitting/parametric.rs

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

    Reference:
        Boom: https://github.com/boom-astro/boom/blob/main/src/fitting/parametric.rs

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

    Reference:
        Boom: https://github.com/boom-astro/boom/blob/main/src/fitting/parametric.rs

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

    Reference:
        Boom: https://github.com/boom-astro/boom/blob/main/src/fitting/parametric.rs

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
