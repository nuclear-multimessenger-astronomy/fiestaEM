"""SALT3 spectral-template supernova model via jax-bandflux.

Uses ``jax_supernovae`` (PyPI: ``jax-bandflux``) for JAX-native, JIT-compiled,
differentiable SALT3 light-curve evaluation.  Unlike the physics-based models
that compute L_bol + R_phot -> blackbody SED, SALT3 uses spectral templates
(M0, M1, colour law) to compute per-band fluxes directly.

The ``jax_supernovae`` import is kept lazy to avoid loading heavy dependencies
for users who don't use SALT3.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array


class SALT3Model:
    """SALT3 spectral-template model for Type Ia supernova light curves.

    Parameters
    ----------
    filters : list[str]
        Band names recognised by ``jax_supernovae.bandpasses`` (e.g.
        ``"ztfg"``, ``"ztfr"``, ``"bessellb"``).
    times : Array, optional
        Observer-frame times (days) at which to evaluate the model.
    redshift : float
        Source redshift (fixed, not sampled).

    Sampled parameters (passed via ``predict(x)``):
        log10_x0 – log10 of the SALT3 amplitude parameter *x0*
        x1       – SALT3 stretch
        c        – SALT3 colour
        t0       – time of B-band maximum (days, same frame as *times*)
    """

    parameter_names: list[str]
    filters: list[str]
    times: Array

    def __init__(self, filters: list[str], times: Array = None,
                 redshift: float = 0.0):
        # Lazy import to avoid loading heavy dependencies at module level
        try:
            from jax_supernovae.salt3 import (
                optimized_salt3_multiband_flux,
                precompute_bandflux_bridge,
            )
            from jax_supernovae.bandpasses import get_bandpass, register_all_bandpasses
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SALT3Model requires the 'jax-bandflux' package. "
                "Install it with: pip install jax-bandflux"
            ) from exc

        register_all_bandpasses()

        if isinstance(filters, str):
            filters = [filters]
        if not filters:
            raise ValueError("At least one filter must be provided")
        self.filters = list(filters)

        if times is None:
            raise ValueError(
                "times must be provided (observer-frame days array)"
            )
        self.times = jnp.asarray(times)
        self.redshift = redshift
        self.parameter_names = ["log10_x0", "x1", "c", "t0"]

        # Pre-compute bridges (expensive, done once)
        self._bridges = tuple(
            precompute_bandflux_bridge(get_bandpass(f)) for f in self.filters
        )
        # Cache zpbandflux_ab per band for AB mag conversion
        self._zpbandflux_ab = jnp.array(
            [b['zpbandflux_ab'] for b in self._bridges]
        )
        # Store reference to the flux function
        self._multiband_flux = optimized_salt3_multiband_flux

    @partial(jax.jit, static_argnums=(0,))
    def predict(self, x: dict[str, Array]) -> tuple[Array, dict[str, Array]]:
        t_days = self.times  # observer-frame days

        # Build SALT3 params dict (z and t0 handled internally by the function)
        x0 = jnp.power(10.0, x["log10_x0"])
        salt3_params = {
            "x0": x0,
            "x1": x["x1"],
            "c": x["c"],
            "z": jnp.array(self.redshift),
            "t0": x["t0"],
        }

        # Compute fluxes: (n_times, n_bands) in photons/s/cm^2
        flux_matrix = self._multiband_flux(t_days, self._bridges, salt3_params)

        # Convert to apparent AB magnitudes
        mag_app = {}
        for i, fname in enumerate(self.filters):
            flux = flux_matrix[:, i]
            mag = -2.5 * jnp.log10(
                jnp.maximum(flux / self._zpbandflux_ab[i], 1e-30)
            )
            mag_app[fname] = mag

        return t_days, mag_app
