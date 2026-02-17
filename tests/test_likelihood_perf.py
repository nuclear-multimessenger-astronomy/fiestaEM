"""Tests for likelihood performance optimizations: JIT compilation,
pre-computed interpolation, and filter constant pre-computation."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from fiesta.inference.lightcurve_model import AfterglowFlux, BullaLightcurveModel
from fiesta.inference.likelihood import EMLikelihood


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LC_FILTERS = ["besselli", "bessellr"]
FLUX_FILTERS = ["bessellv"]

BU_PARAMS = [120, -2, 0.2, 0.3, -1.5, 0.4, 0.3, 0.1]
GRB_PARAMS = [3.141 / 30, 54.0, 0.05, 2.0, -1.0, 2.5, -2.0, -4.0, 500]


def _make_synthetic_data(filters, trigger_time=60000.0, n_obs=10, n_nondet=2, seed=42):
    """Create synthetic photometry data for testing."""
    rng = np.random.RandomState(seed)
    data = {}
    for filt in filters:
        times = trigger_time + np.sort(rng.uniform(0.5, 10.0, n_obs))
        mags = rng.uniform(18, 22, n_obs)
        mag_errs = rng.uniform(0.1, 0.5, n_obs)
        mag_errs[-n_nondet:] = np.inf  # non-detections
        data[filt] = np.column_stack([times, mags, mag_errs])
    return data


def _make_lc_likelihood():
    """Build a LightcurveModel-based likelihood (uses precomputed interp)."""
    model = BullaLightcurveModel(name="Bu2025", filters=LC_FILTERS)
    data = _make_synthetic_data(model.filters)
    lk = EMLikelihood(model=model, data=data, trigger_time=60000.0,
                      tmin=0.1, tmax=15.0)
    params = dict(zip(model.parameter_names, BU_PARAMS))
    params["luminosity_distance"] = 40.0
    params["redshift"] = 0.0
    return lk, params


def _make_flux_likelihood():
    """Build a FluxModel-based likelihood (falls back to jnp.interp)."""
    model = AfterglowFlux(name="pbag_gaussian_MLP", filters=FLUX_FILTERS)
    data = _make_synthetic_data(model.filters)
    lk = EMLikelihood(model=model, data=data, trigger_time=60000.0,
                      tmin=0.1, tmax=15.0)
    params = dict(zip(model.parameter_names, GRB_PARAMS))
    params["luminosity_distance"] = 40.0
    params["redshift"] = 0.01
    return lk, params


# ---------------------------------------------------------------------------
# 1. Pre-computed interpolation
# ---------------------------------------------------------------------------

class TestPrecomputedInterpolation:

    def test_enabled_for_lightcurve_model(self):
        lk, _ = _make_lc_likelihood()
        assert lk._use_precomputed_interp is True

    def test_disabled_for_flux_model(self):
        lk, _ = _make_flux_likelihood()
        assert lk._use_precomputed_interp is False

    def test_weights_match_jnp_interp(self):
        """Pre-computed interpolation must match jnp.interp for interior points."""
        model = BullaLightcurveModel(name="Bu2025", filters=LC_FILTERS)
        model_times = jnp.asarray(model.times)

        obs_times = jnp.array([1.0, 3.0, 5.0, 10.0])
        values = jnp.sin(model_times)  # arbitrary test signal

        # Pre-computed path
        idx, w = EMLikelihood._compute_interp_weights(obs_times, model_times)
        precomp = EMLikelihood._apply_precomputed_interp(idx, w, values)

        # Reference: jnp.interp
        reference = jnp.interp(obs_times, model_times, values,
                               left="extrapolate", right="extrapolate")

        np.testing.assert_allclose(precomp, reference, rtol=1e-5)

    def test_extrapolation_matches(self):
        """Pre-computed weights must extrapolate consistently with jnp.interp."""
        model_times = jnp.linspace(1.0, 10.0, 50)
        values = jnp.linspace(20.0, 25.0, 50)
        obs_times = jnp.array([0.1, 0.5, 11.0, 15.0])  # outside range

        idx, w = EMLikelihood._compute_interp_weights(obs_times, model_times)
        precomp = EMLikelihood._apply_precomputed_interp(idx, w, values)
        reference = jnp.interp(obs_times, model_times, values,
                               left="extrapolate", right="extrapolate")

        np.testing.assert_allclose(precomp, reference, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. JIT-compiled evaluate
# ---------------------------------------------------------------------------

class TestJITEvaluate:

    def test_lc_evaluate_returns_finite(self):
        lk, params = _make_lc_likelihood()
        result = lk.evaluate(params)
        assert jnp.isfinite(result), f"Expected finite, got {result}"

    def test_flux_evaluate_returns_finite(self):
        lk, params = _make_flux_likelihood()
        result = lk.evaluate(params)
        assert jnp.isfinite(result), f"Expected finite, got {result}"

    def test_repeated_calls_consistent(self):
        """JIT cache must return identical results across calls."""
        lk, params = _make_lc_likelihood()
        r1 = lk.evaluate(params)
        r2 = lk.evaluate(params)
        np.testing.assert_allclose(r1, r2)

    def test_call_matches_evaluate(self):
        lk, params = _make_lc_likelihood()
        assert jnp.allclose(lk(params), lk.evaluate(params))

    def test_grad_through_evaluate(self):
        """Gradients must flow through the JIT-compiled likelihood."""
        lk, params = _make_lc_likelihood()
        # jax.grad requires float-valued inputs
        params = {k: jnp.float32(v) for k, v in params.items()}
        grad_fn = jax.grad(lambda p: lk.evaluate(p))
        grads = grad_fn(params)
        # Check at least one gradient is non-zero and finite
        any_nonzero = any(
            jnp.any(jnp.abs(v) > 0) for v in jax.tree_util.tree_leaves(grads)
        )
        all_finite = all(
            jnp.all(jnp.isfinite(v)) for v in jax.tree_util.tree_leaves(grads)
        )
        assert all_finite, "Gradients contain non-finite values"
        assert any_nonzero, "All gradients are zero"


# ---------------------------------------------------------------------------
# 3. Chunked v_evaluate
# ---------------------------------------------------------------------------

class TestVEvaluate:

    def test_batch_matches_single(self):
        lk, params = _make_lc_likelihood()
        single = lk.evaluate(params)
        batch_params = {k: jnp.array([v, v, v]) for k, v in params.items()}
        batch = lk.v_evaluate(batch_params)
        assert batch.shape == (3,)
        np.testing.assert_allclose(batch, single, rtol=1e-5)

    def test_chunked_matches_unchunked(self):
        lk, params = _make_lc_likelihood()
        batch_params = {k: jnp.tile(v, 5) for k, v in params.items()}
        full = lk.v_evaluate(batch_params, chunk_size=10000)
        chunked = lk.v_evaluate(batch_params, chunk_size=2)
        np.testing.assert_allclose(chunked, full, rtol=1e-5)


# ---------------------------------------------------------------------------
# 4. Filter constant pre-computation
# ---------------------------------------------------------------------------

class TestFilterPrecomputation:

    def test_bandpass_has_precomputed_log(self):
        from fiesta.filters import Filter
        f = Filter("bessellv")
        assert hasattr(f, "_log_trans_over_hnu")
        assert f._log_trans_over_hnu.shape == f.nus.shape

    def test_bandpass_mag_unchanged(self):
        """Pre-computed constants must produce identical magnitudes."""
        from fiesta.conversions import bandpass_AB_mag
        from fiesta.filters import Filter
        import fiesta.constants as constants

        filt = Filter("bessellr")
        # Synthetic flux: shape (n_nus, n_times)
        rng = np.random.RandomState(0)
        nus = jnp.linspace(filt.nus[0] * 0.8, filt.nus[-1] * 1.2, 200)
        flux = jnp.abs(jnp.array(rng.randn(200, 10))) * 1e3 + 1.0

        # With precomputed
        mag_precomp = bandpass_AB_mag(flux, nus, filt.nus, filt.trans,
                                      filt.ref_flux, filt._log_trans_over_hnu)
        # Without precomputed (original path)
        mag_orig = bandpass_AB_mag(flux, nus, filt.nus, filt.trans,
                                   filt.ref_flux, None)

        np.testing.assert_allclose(mag_precomp, mag_orig, rtol=1e-5)
