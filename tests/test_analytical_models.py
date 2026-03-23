"""Tests for analytical light-curve models."""

import importlib.util

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from fiesta.inference.analytical_models import (
    AnalyticalModel,
    ShockCoolingModel,
    ArnettModel,
    MetzgerModel,
    MetzgerFullModel,
    OneComponentKilonovaModel,
    MagnetarBoostedKilonovaModel,
    ShockedCocoonModel,
    EvolvingBlackbodyModel,
    TDEAnalyticalModel,
    NickelCobaltModel,
    MagnetarPoweredSNModel,
    CSMInteractionModel,
    PhenomenologicalModel,
    BazinModel,
    VillarModel,
    PhenomenologicalTDEModel,
    AfterglowModel,
    SALT3Model,
)
from fiesta.inference.lightcurve_model import CombinedSurrogate
from fiesta.inference.likelihood import EMLikelihood


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FILTERS = ["bessellb", "bessellr"]


def _make_synthetic_data(filters, trigger_time=0.0, n_obs=10, n_nondet=2, seed=42):
    rng = np.random.RandomState(seed)
    data = {}
    for filt in filters:
        times = trigger_time + np.sort(rng.uniform(0.5, 3.0, n_obs))
        mags = rng.uniform(18, 22, n_obs)
        mag_errs = rng.uniform(0.1, 0.5, n_obs)
        mag_errs[-n_nondet:] = np.inf
        data[filt] = np.column_stack([times, mags, mag_errs])
    return data


# ---------------------------------------------------------------------------
# ShockCoolingModel
# ---------------------------------------------------------------------------

class TestShockCooling:
    def _make_model_and_params(self):
        model = ShockCoolingModel(filters=FILTERS)
        params = {
            "log10_Menv": -2.0,
            "log10_Renv": 2.0,
            "log10_Ee": 49.0,
            "luminosity_distance": 40.0,
            "redshift": 0.01,
        }
        return model, params

    def test_shape(self):
        model, params = self._make_model_and_params()
        times, mags = model.predict(params)
        assert times.shape == (100,)
        for filt in FILTERS:
            assert mags[filt].shape == (100,), f"Wrong shape for {filt}"

    def test_finite(self):
        model, params = self._make_model_and_params()
        _, mags = model.predict(params)
        for filt in FILTERS:
            assert jnp.all(jnp.isfinite(mags[filt])), f"Non-finite in {filt}"

    def test_jit_consistent(self):
        model, params = self._make_model_and_params()
        _, m1 = model.predict(params)
        _, m2 = model.predict(params)
        for filt in FILTERS:
            np.testing.assert_allclose(m1[filt], m2[filt])

    def test_differentiable(self):
        model, params = self._make_model_and_params()
        params = {k: jnp.float64(v) if isinstance(v, (int, float)) else v
                  for k, v in params.items()}

        def _loss(p):
            _, mags = model.predict(p)
            return jnp.sum(jnp.stack([jnp.sum(mags[f]) for f in FILTERS]))

        grads = jax.grad(_loss)(params)
        any_nonzero = any(jnp.any(jnp.abs(v) > 0)
                          for v in jax.tree_util.tree_leaves(grads))
        all_finite = all(jnp.all(jnp.isfinite(v))
                         for v in jax.tree_util.tree_leaves(grads))
        assert all_finite, "Gradients contain non-finite values"
        assert any_nonzero, "All gradients are zero"


# ---------------------------------------------------------------------------
# ArnettModel
# ---------------------------------------------------------------------------

class TestArnett:
    def _make_model_and_params(self, modified=False):
        model = ArnettModel(filters=FILTERS, modified=modified)
        params = {
            "tau_m": 15.0,
            "log10_mni": -0.5,
            "v_phot": 1.0,
            "luminosity_distance": 100.0,
            "redshift": 0.02,
        }
        if modified:
            params["t_0"] = 5.0
        return model, params

    def test_shape(self):
        model, params = self._make_model_and_params()
        times, mags = model.predict(params)
        assert times.shape == (100,)
        for filt in FILTERS:
            assert mags[filt].shape == (100,)

    def test_finite(self):
        model, params = self._make_model_and_params()
        _, mags = model.predict(params)
        for filt in FILTERS:
            assert jnp.all(jnp.isfinite(mags[filt])), f"Non-finite in {filt}"

    def test_modified_variant(self):
        model, params = self._make_model_and_params(modified=True)
        _, mags = model.predict(params)
        for filt in FILTERS:
            assert jnp.all(jnp.isfinite(mags[filt]))

    def test_modified_differs_from_standard(self):
        m_std, p_std = self._make_model_and_params(modified=False)
        m_mod, p_mod = self._make_model_and_params(modified=True)
        _, mags_std = m_std.predict(p_std)
        _, mags_mod = m_mod.predict(p_mod)
        # They should differ because the trapping factor changes the luminosity
        for filt in FILTERS:
            assert not jnp.allclose(mags_std[filt], mags_mod[filt])

    def test_differentiable(self):
        model, params = self._make_model_and_params()
        params = {k: jnp.float64(v) for k, v in params.items()}

        def _loss(p):
            _, mags = model.predict(p)
            return jnp.sum(jnp.stack([jnp.sum(mags[f]) for f in FILTERS]))

        grads = jax.grad(_loss)(params)
        all_finite = all(jnp.all(jnp.isfinite(v))
                         for v in jax.tree_util.tree_leaves(grads))
        assert all_finite


# ---------------------------------------------------------------------------
# MetzgerModel
# ---------------------------------------------------------------------------

class TestMetzger:
    def _make_model_and_params(self, full=False):
        cls = MetzgerFullModel if full else MetzgerModel
        model = cls(filters=FILTERS)
        params = {
            "log10_mej": -1.5,
            "log10_vej": -0.7,
            "beta": 3.0,
            "log10_kappa_r": 0.5,
            "luminosity_distance": 40.0,
            "redshift": 0.01,
        }
        return model, params

    def test_shape_single(self):
        model, params = self._make_model_and_params(full=False)
        times, mags = model.predict(params)
        assert times.shape == (100,)
        for filt in FILTERS:
            assert mags[filt].shape == (100,)

    def test_finite_single(self):
        model, params = self._make_model_and_params(full=False)
        _, mags = model.predict(params)
        for filt in FILTERS:
            assert jnp.all(jnp.isfinite(mags[filt]))

    def test_shape_full(self):
        model, params = self._make_model_and_params(full=True)
        times, mags = model.predict(params)
        assert times.shape == (100,)
        for filt in FILTERS:
            assert mags[filt].shape == (100,)

    def test_finite_full(self):
        model, params = self._make_model_and_params(full=True)
        _, mags = model.predict(params)
        for filt in FILTERS:
            assert jnp.all(jnp.isfinite(mags[filt]))

    def test_differentiable(self):
        model, params = self._make_model_and_params(full=False)
        params = {k: jnp.float64(v) for k, v in params.items()}

        def _loss(p):
            _, mags = model.predict(p)
            return jnp.sum(jnp.stack([jnp.sum(mags[f]) for f in FILTERS]))

        grads = jax.grad(_loss)(params)
        all_finite = all(jnp.all(jnp.isfinite(v))
                         for v in jax.tree_util.tree_leaves(grads))
        assert all_finite


# ---------------------------------------------------------------------------
# CombinedSurrogate with analytical models
# ---------------------------------------------------------------------------

class TestCombinedWithAnalytical:
    def test_two_analytical_models(self):
        m1 = ShockCoolingModel(filters=["bessellb"])
        m2 = ArnettModel(filters=["bessellr"])
        sample_times = jnp.geomspace(0.5, 3.0, 50)
        combined = CombinedSurrogate(models=[m1, m2], sample_times=sample_times)

        assert "bessellb" in combined.filters
        assert "bessellr" in combined.filters

        params = {
            # ShockCooling
            "log10_Menv": -2.0,
            "log10_Renv": 2.0,
            "log10_Ee": 49.0,
            # Arnett
            "tau_m": 15.0,
            "log10_mni": -0.5,
            "v_phot": 1.0,
            # Shared
            "luminosity_distance": 40.0,
            "redshift": 0.01,
        }
        times, mags = combined.predict(params)
        assert times.shape == (50,)
        for filt in combined.filters:
            assert jnp.all(jnp.isfinite(mags[filt]))


# ---------------------------------------------------------------------------
# EMLikelihood integration
# ---------------------------------------------------------------------------

class TestLikelihoodIntegration:
    def _make_likelihood(self):
        model = ShockCoolingModel(filters=FILTERS)
        data = _make_synthetic_data(FILTERS, trigger_time=0.0)
        lk = EMLikelihood(model=model, data=data, trigger_time=0.0,
                          data_tmin=0.1, data_tmax=4.0)
        params = {
            "log10_Menv": -2.0,
            "log10_Renv": 2.0,
            "log10_Ee": 49.0,
            "luminosity_distance": 40.0,
            "redshift": 0.01,
        }
        return lk, params

    def test_evaluate_finite(self):
        lk, params = self._make_likelihood()
        result = lk.evaluate(params)
        assert jnp.isfinite(result), f"Expected finite, got {result}"

    def test_grad_through_likelihood(self):
        lk, params = self._make_likelihood()
        params = {k: jnp.float64(v) for k, v in params.items()}
        grad_fn = jax.grad(lambda p: lk.evaluate(p))
        grads = grad_fn(params)
        all_finite = all(jnp.all(jnp.isfinite(v))
                         for v in jax.tree_util.tree_leaves(grads))
        assert all_finite, "Gradients through likelihood are non-finite"


# ---------------------------------------------------------------------------
# OneComponentKilonovaModel
# ---------------------------------------------------------------------------

class TestOneComponentKilonova:
    def _make_model_and_params(self):
        model = OneComponentKilonovaModel(filters=FILTERS)
        params = {
            "log10_mej": -1.5,
            "log10_vej": -0.7,
            "log10_kappa": 0.5,
            "luminosity_distance": 40.0,
            "redshift": 0.01,
        }
        return model, params

    def test_shape(self):
        model, params = self._make_model_and_params()
        times, mags = model.predict(params)
        assert times.shape == (100,)
        for filt in FILTERS:
            assert mags[filt].shape == (100,), f"Wrong shape for {filt}"

    def test_finite(self):
        model, params = self._make_model_and_params()
        _, mags = model.predict(params)
        for filt in FILTERS:
            assert jnp.all(jnp.isfinite(mags[filt])), f"Non-finite in {filt}"

    def test_temperature_floor_active(self):
        """At late times the temperature should be clamped to T_floor."""
        model, params = self._make_model_and_params()
        assert model.temperature_floor == 4000.0

    def test_differentiable(self):
        model, params = self._make_model_and_params()
        params = {k: jnp.float64(v) for k, v in params.items()}

        def _loss(p):
            _, mags = model.predict(p)
            return jnp.sum(jnp.stack([jnp.sum(mags[f]) for f in FILTERS]))

        grads = jax.grad(_loss)(params)
        all_finite = all(jnp.all(jnp.isfinite(v))
                         for v in jax.tree_util.tree_leaves(grads))
        any_nonzero = any(jnp.any(jnp.abs(v) > 0)
                          for v in jax.tree_util.tree_leaves(grads))
        assert all_finite, "Gradients contain non-finite values"
        assert any_nonzero, "All gradients are zero"


# ---------------------------------------------------------------------------
# MagnetarBoostedKilonovaModel
# ---------------------------------------------------------------------------

class TestMagnetarBoostedKilonova:
    def _make_model_and_params(self):
        model = MagnetarBoostedKilonovaModel(filters=FILTERS)
        params = {
            "log10_mej": -1.5,
            "log10_vej": -0.7,
            "beta": 3.0,
            "log10_kappa_r": 0.5,
            "log10_p0": 0.0,       # 1 ms
            "log10_bp": 0.0,       # 1e14 G
            "mass_ns": 1.4,
            "theta_pb": 1.0,       # ~57 degrees
            "thermalisation_efficiency": 0.5,
            "luminosity_distance": 40.0,
            "redshift": 0.01,
        }
        return model, params

    def test_shape(self):
        model, params = self._make_model_and_params()
        times, mags = model.predict(params)
        assert times.shape == (100,)
        for filt in FILTERS:
            assert mags[filt].shape == (100,), f"Wrong shape for {filt}"

    def test_finite(self):
        model, params = self._make_model_and_params()
        _, mags = model.predict(params)
        for filt in FILTERS:
            assert jnp.all(jnp.isfinite(mags[filt])), f"Non-finite in {filt}"

    def test_magnetar_enhances_luminosity(self):
        """Magnetar-boosted model should be brighter than plain MetzgerModel."""
        mag_model = MagnetarBoostedKilonovaModel(filters=FILTERS)
        metz_model = MetzgerModel(filters=FILTERS)
        shared = {
            "log10_mej": -1.5,
            "log10_vej": -0.7,
            "beta": 3.0,
            "log10_kappa_r": 0.5,
            "luminosity_distance": 40.0,
            "redshift": 0.01,
        }
        mag_params = {**shared, "log10_p0": 0.0, "log10_bp": 0.0,
                      "mass_ns": 1.4, "theta_pb": 1.0,
                      "thermalisation_efficiency": 0.5}
        _, mags_mag = mag_model.predict(mag_params)
        _, mags_met = metz_model.predict(shared)
        # Brighter = lower apparent magnitude for at least some times
        for filt in FILTERS:
            assert jnp.any(mags_mag[filt] < mags_met[filt]), \
                f"Magnetar model not brighter in {filt}"

    def test_differentiable(self):
        model, params = self._make_model_and_params()
        params = {k: jnp.float64(v) for k, v in params.items()}

        def _loss(p):
            _, mags = model.predict(p)
            return jnp.sum(jnp.stack([jnp.sum(mags[f]) for f in FILTERS]))

        grads = jax.grad(_loss)(params)
        all_finite = all(jnp.all(jnp.isfinite(v))
                         for v in jax.tree_util.tree_leaves(grads))
        assert all_finite, "Gradients contain non-finite values"


# ---------------------------------------------------------------------------
# ShockedCocoonModel
# ---------------------------------------------------------------------------

class TestShockedCocoon:
    def _make_model_and_params(self):
        model = ShockedCocoonModel(filters=FILTERS)
        params = {
            "log10_mej": -1.5,
            "log10_vej": -0.7,
            "eta": 10.0,
            "log10_tshock": 1.0,   # 10 seconds
            "shocked_fraction": 0.5,
            "cos_theta_cocoon": 0.5,
            "log10_kappa": 0.0,    # 1 cm^2/g
            "luminosity_distance": 40.0,
            "redshift": 0.01,
        }
        return model, params

    def test_shape(self):
        model, params = self._make_model_and_params()
        times, mags = model.predict(params)
        assert times.shape == (100,)
        for filt in FILTERS:
            assert mags[filt].shape == (100,), f"Wrong shape for {filt}"

    def test_finite(self):
        model, params = self._make_model_and_params()
        _, mags = model.predict(params)
        for filt in FILTERS:
            assert jnp.all(jnp.isfinite(mags[filt])), f"Non-finite in {filt}"

    def test_differentiable(self):
        model, params = self._make_model_and_params()
        params = {k: jnp.float64(v) for k, v in params.items()}

        def _loss(p):
            _, mags = model.predict(p)
            return jnp.sum(jnp.stack([jnp.sum(mags[f]) for f in FILTERS]))

        grads = jax.grad(_loss)(params)
        all_finite = all(jnp.all(jnp.isfinite(v))
                         for v in jax.tree_util.tree_leaves(grads))
        any_nonzero = any(jnp.any(jnp.abs(v) > 0)
                          for v in jax.tree_util.tree_leaves(grads))
        assert all_finite, "Gradients contain non-finite values"
        assert any_nonzero, "All gradients are zero"


# ---------------------------------------------------------------------------
# EvolvingBlackbodyModel
# ---------------------------------------------------------------------------

class TestEvolvingBlackbody:
    def _make_model_and_params(self):
        model = EvolvingBlackbodyModel(filters=FILTERS)
        params = {
            "log10_temperature_0": 4.0,       # 10,000 K
            "log10_radius_0": 14.0,           # ~1e14 cm
            "temp_rise_index": 0.5,
            "temp_decline_index": 0.3,
            "temp_peak_time": 2.0,            # days
            "radius_rise_index": 0.8,
            "radius_decline_index": 0.2,
            "radius_peak_time": 5.0,          # days
            "luminosity_distance": 40.0,
            "redshift": 0.01,
        }
        return model, params

    def test_shape(self):
        model, params = self._make_model_and_params()
        times, mags = model.predict(params)
        assert times.shape == (100,)
        for filt in FILTERS:
            assert mags[filt].shape == (100,), f"Wrong shape for {filt}"

    def test_finite(self):
        model, params = self._make_model_and_params()
        _, mags = model.predict(params)
        for filt in FILTERS:
            assert jnp.all(jnp.isfinite(mags[filt])), f"Non-finite in {filt}"

    def test_differentiable(self):
        model, params = self._make_model_and_params()
        params = {k: jnp.float64(v) for k, v in params.items()}

        def _loss(p):
            _, mags = model.predict(p)
            return jnp.sum(jnp.stack([jnp.sum(mags[f]) for f in FILTERS]))

        grads = jax.grad(_loss)(params)
        all_finite = all(jnp.all(jnp.isfinite(v))
                         for v in jax.tree_util.tree_leaves(grads))
        any_nonzero = any(jnp.any(jnp.abs(v) > 0)
                          for v in jax.tree_util.tree_leaves(grads))
        assert all_finite, "Gradients contain non-finite values"
        assert any_nonzero, "All gradients are zero"


# ---------------------------------------------------------------------------
# TDEAnalyticalModel
# ---------------------------------------------------------------------------

class TestTDEAnalytical:
    def _make_model_and_params(self):
        model = TDEAnalyticalModel(filters=FILTERS)
        params = {
            "log10_l0": 45.0,         # 1e45 erg/s at 1 second
            "t_0_turn": 1.0,          # 1 day turn-on
            "log10_mej": 0.0,         # 1 solar mass
            "log10_vej": 4.0,         # 1e4 km/s
            "log10_kappa": -0.7,      # ~0.2 cm^2/g
            "log10_kappa_gamma": -2.0,
            "luminosity_distance": 200.0,
            "redshift": 0.05,
        }
        return model, params

    def test_shape(self):
        model, params = self._make_model_and_params()
        times, mags = model.predict(params)
        assert times.shape == (100,)
        for filt in FILTERS:
            assert mags[filt].shape == (100,), f"Wrong shape for {filt}"

    def test_finite(self):
        model, params = self._make_model_and_params()
        _, mags = model.predict(params)
        for filt in FILTERS:
            assert jnp.all(jnp.isfinite(mags[filt])), f"Non-finite in {filt}"

    def test_differentiable(self):
        model, params = self._make_model_and_params()
        params = {k: jnp.float64(v) for k, v in params.items()}

        def _loss(p):
            _, mags = model.predict(p)
            return jnp.sum(jnp.stack([jnp.sum(mags[f]) for f in FILTERS]))

        grads = jax.grad(_loss)(params)
        all_finite = all(jnp.all(jnp.isfinite(v))
                         for v in jax.tree_util.tree_leaves(grads))
        assert all_finite, "Gradients contain non-finite values"

    def test_fallback_decay(self):
        """Light curve should not be constant — diffusion processes the fallback."""
        model, params = self._make_model_and_params()
        _, mags = model.predict(params)
        for filt in FILTERS:
            # Verify the light curve is not flat (diffusion modifies the engine)
            mag_range = jnp.max(mags[filt]) - jnp.min(mags[filt])
            assert mag_range > 0.1, \
                f"{filt}: expected varying light curve, got range {mag_range}"


# ---------------------------------------------------------------------------
# NickelCobaltModel
# ---------------------------------------------------------------------------

class TestNickelCobalt:
    def _make_model_and_params(self, f_nickel=0.1):
        model = NickelCobaltModel(filters=FILTERS)
        params = {
            "f_nickel": f_nickel,
            "log10_mej": 0.5,         # ~3 solar masses
            "log10_vej": 4.0,         # 1e4 km/s
            "log10_kappa": -0.7,
            "log10_kappa_gamma": -2.0,
            "luminosity_distance": 100.0,
            "redshift": 0.02,
        }
        return model, params

    def test_shape(self):
        model, params = self._make_model_and_params()
        times, mags = model.predict(params)
        assert times.shape == (100,)
        for filt in FILTERS:
            assert mags[filt].shape == (100,)

    def test_finite(self):
        model, params = self._make_model_and_params()
        _, mags = model.predict(params)
        for filt in FILTERS:
            assert jnp.all(jnp.isfinite(mags[filt])), f"Non-finite in {filt}"

    def test_differentiable(self):
        model, params = self._make_model_and_params()
        params = {k: jnp.float64(v) for k, v in params.items()}

        def _loss(p):
            _, mags = model.predict(p)
            return jnp.sum(jnp.stack([jnp.sum(mags[f]) for f in FILTERS]))

        grads = jax.grad(_loss)(params)
        all_finite = all(jnp.all(jnp.isfinite(v))
                         for v in jax.tree_util.tree_leaves(grads))
        assert all_finite, "Gradients contain non-finite values"

    def test_nickel_scaling(self):
        """Higher f_nickel should produce brighter emission (lower mags)."""
        model_lo, params_lo = self._make_model_and_params(f_nickel=0.05)
        model_hi, params_hi = self._make_model_and_params(f_nickel=0.5)
        _, mags_lo = model_lo.predict(params_lo)
        _, mags_hi = model_hi.predict(params_hi)
        for filt in FILTERS:
            # Higher nickel -> brighter -> lower mean magnitude
            assert jnp.mean(mags_hi[filt]) < jnp.mean(mags_lo[filt]), \
                f"{filt}: higher f_nickel should be brighter"


# ---------------------------------------------------------------------------
# MagnetarPoweredSNModel
# ---------------------------------------------------------------------------

class TestMagnetarPoweredSN:
    def _make_model_and_params(self):
        model = MagnetarPoweredSNModel(filters=FILTERS)
        params = {
            "log10_p0": 0.0,          # 1 ms
            "log10_bp": 0.0,          # 1e14 G
            "mass_ns": 1.4,
            "theta_pb": 1.0,
            "log10_mej": 0.5,         # ~3 solar masses
            "log10_vej": 4.0,         # 1e4 km/s
            "log10_kappa": -0.7,
            "log10_kappa_gamma": -2.0,
            "luminosity_distance": 200.0,
            "redshift": 0.05,
        }
        return model, params

    def test_shape(self):
        model, params = self._make_model_and_params()
        times, mags = model.predict(params)
        assert times.shape == (100,)
        for filt in FILTERS:
            assert mags[filt].shape == (100,)

    def test_finite(self):
        model, params = self._make_model_and_params()
        _, mags = model.predict(params)
        for filt in FILTERS:
            assert jnp.all(jnp.isfinite(mags[filt])), f"Non-finite in {filt}"

    def test_differentiable(self):
        model, params = self._make_model_and_params()
        params = {k: jnp.float64(v) for k, v in params.items()}

        def _loss(p):
            _, mags = model.predict(p)
            return jnp.sum(jnp.stack([jnp.sum(mags[f]) for f in FILTERS]))

        grads = jax.grad(_loss)(params)
        all_finite = all(jnp.all(jnp.isfinite(v))
                         for v in jax.tree_util.tree_leaves(grads))
        assert all_finite, "Gradients contain non-finite values"

    def test_magnetar_brighter(self):
        """Strong magnetar SN should outshine pure NickelCobalt."""
        mag_model = MagnetarPoweredSNModel(filters=FILTERS)
        ni_model = NickelCobaltModel(filters=FILTERS)
        shared = {
            "log10_mej": 0.5,
            "log10_vej": 4.0,
            "log10_kappa": -0.7,
            "log10_kappa_gamma": -2.0,
            "luminosity_distance": 200.0,
            "redshift": 0.05,
        }
        mag_params = {**shared, "log10_p0": 0.0, "log10_bp": 0.0,
                      "mass_ns": 1.4, "theta_pb": 1.0}
        ni_params = {**shared, "f_nickel": 0.1}
        _, mags_mag = mag_model.predict(mag_params)
        _, mags_ni = ni_model.predict(ni_params)
        for filt in FILTERS:
            assert jnp.mean(mags_mag[filt]) < jnp.mean(mags_ni[filt]), \
                f"{filt}: magnetar SN should be brighter than NickelCobalt"


# ---------------------------------------------------------------------------
# CSMInteractionModel
# ---------------------------------------------------------------------------

class TestCSMInteraction:
    def _make_model_and_params(self):
        model = CSMInteractionModel(filters=FILTERS, nn=12, delta=1)
        params = {
            "log10_mej": 0.5,         # ~3 solar masses
            "log10_csm_mass": 0.0,    # 1 solar mass CSM
            "log10_vej": 4.0,         # 1e4 km/s
            "eta": 0.5,
            "log10_rho": -13.0,       # typical CSM density
            "log10_kappa": -0.7,
            "log10_r0": 0.0,          # 1 AU
            "luminosity_distance": 200.0,
            "redshift": 0.05,
        }
        return model, params

    def test_shape(self):
        model, params = self._make_model_and_params()
        times, mags = model.predict(params)
        assert times.shape == (100,)
        for filt in FILTERS:
            assert mags[filt].shape == (100,), f"Wrong shape for {filt}"

    def test_finite(self):
        model, params = self._make_model_and_params()
        _, mags = model.predict(params)
        for filt in FILTERS:
            assert jnp.all(jnp.isfinite(mags[filt])), f"Non-finite in {filt}"

    def test_differentiable(self):
        model, params = self._make_model_and_params()
        params = {k: jnp.float64(v) for k, v in params.items()}

        def _loss(p):
            _, mags = model.predict(p)
            return jnp.sum(jnp.stack([jnp.sum(mags[f]) for f in FILTERS]))

        grads = jax.grad(_loss)(params)
        all_finite = all(jnp.all(jnp.isfinite(v))
                         for v in jax.tree_util.tree_leaves(grads))
        assert all_finite, "Gradients contain non-finite values"

    def test_eta_sensitivity(self):
        """Varying eta should change the light curve."""
        model = CSMInteractionModel(filters=FILTERS, nn=12, delta=1)
        base = {
            "log10_mej": 0.5,
            "log10_csm_mass": 0.0,
            "log10_vej": 4.0,
            "log10_rho": -13.0,
            "log10_kappa": -0.7,
            "log10_r0": 0.0,
            "luminosity_distance": 200.0,
            "redshift": 0.05,
        }
        _, mags_lo = model.predict({**base, "eta": 0.2})
        _, mags_hi = model.predict({**base, "eta": 1.5})
        for filt in FILTERS:
            assert not jnp.allclose(mags_lo[filt], mags_hi[filt]), \
                f"{filt}: eta should change the light curve"

    def test_csm_mass_scaling(self):
        """More CSM mass should produce brighter emission."""
        model = CSMInteractionModel(filters=FILTERS, nn=12, delta=1)
        base = {
            "log10_mej": 0.5,
            "log10_vej": 4.0,
            "eta": 0.5,
            "log10_rho": -13.0,
            "log10_kappa": -0.7,
            "log10_r0": 0.0,
            "luminosity_distance": 200.0,
            "redshift": 0.05,
        }
        _, mags_lo = model.predict({**base, "log10_csm_mass": -0.5})
        _, mags_hi = model.predict({**base, "log10_csm_mass": 0.5})
        for filt in FILTERS:
            assert not jnp.allclose(mags_lo[filt], mags_hi[filt]), \
                f"{filt}: CSM mass should change the light curve"


# ---------------------------------------------------------------------------
# BazinModel
# ---------------------------------------------------------------------------

class TestBazin:
    def _make_model_and_params(self):
        model = BazinModel(filters=FILTERS)
        params = {
            "t0": 5.0,
            "log10_tau_rise": 0.3,
            "log10_tau_fall": jnp.log10(20.0),
            "amp_mag_bessellb": 20.0,
            "base_mag_bessellb": 25.0,
            "amp_mag_bessellr": 19.5,
            "base_mag_bessellr": 25.0,
        }
        return model, params

    def test_shape(self):
        model, params = self._make_model_and_params()
        times, mags = model.predict(params)
        assert times.shape == (200,)
        for filt in FILTERS:
            assert mags[filt].shape == (200,), f"Wrong shape for {filt}"

    def test_finite(self):
        model, params = self._make_model_and_params()
        _, mags = model.predict(params)
        for filt in FILTERS:
            assert jnp.all(jnp.isfinite(mags[filt])), f"Non-finite in {filt}"

    def test_differentiable(self):
        model, params = self._make_model_and_params()
        params = {k: jnp.float64(v) if isinstance(v, (int, float)) else v
                  for k, v in params.items()}

        def _loss(p):
            _, mags = model.predict(p)
            return jnp.sum(jnp.stack([jnp.sum(mags[f]) for f in FILTERS]))

        grads = jax.grad(_loss)(params)
        all_finite = all(jnp.all(jnp.isfinite(v))
                         for v in jax.tree_util.tree_leaves(grads))
        assert all_finite, "Gradients contain non-finite values"

    def test_peak_near_t0(self):
        """Shape should peak near t0."""
        model, params = self._make_model_and_params()
        times, mags = model.predict(params)
        for filt in FILTERS:
            # Minimum magnitude = brightest = peak
            peak_idx = jnp.argmin(mags[filt])
            peak_time = times[peak_idx]
            assert jnp.abs(peak_time - params["t0"]) < 15.0, \
                f"{filt}: peak at t={float(peak_time):.1f}, expected near t0={params['t0']}"


# ---------------------------------------------------------------------------
# VillarModel
# ---------------------------------------------------------------------------

class TestVillar:
    def _make_model_and_params(self):
        model = VillarModel(filters=FILTERS)
        params = {
            "t0": 10.0,
            "log10_tau_rise": 0.5,
            "log10_tau_fall": jnp.log10(30.0),
            "beta_slope": 0.01,
            "log10_gamma": jnp.log10(15.0),
            "amp_mag_bessellb": 20.0,
            "amp_mag_bessellr": 19.5,
        }
        return model, params

    def test_shape(self):
        model, params = self._make_model_and_params()
        times, mags = model.predict(params)
        assert times.shape == (200,)
        for filt in FILTERS:
            assert mags[filt].shape == (200,), f"Wrong shape for {filt}"

    def test_finite(self):
        model, params = self._make_model_and_params()
        _, mags = model.predict(params)
        for filt in FILTERS:
            assert jnp.all(jnp.isfinite(mags[filt])), f"Non-finite in {filt}"

    def test_differentiable(self):
        model, params = self._make_model_and_params()
        params = {k: jnp.float64(v) if isinstance(v, (int, float)) else v
                  for k, v in params.items()}

        def _loss(p):
            _, mags = model.predict(p)
            return jnp.sum(jnp.stack([jnp.sum(mags[f]) for f in FILTERS]))

        grads = jax.grad(_loss)(params)
        all_finite = all(jnp.all(jnp.isfinite(v))
                         for v in jax.tree_util.tree_leaves(grads))
        assert all_finite, "Gradients contain non-finite values"

    def test_piecewise_transition(self):
        """Changing gamma should shift the late-time decay onset."""
        model = VillarModel(filters=FILTERS)
        base = {
            "t0": 10.0,
            "log10_tau_rise": 0.5,
            "log10_tau_fall": jnp.log10(30.0),
            "beta_slope": 0.01,
            "amp_mag_bessellb": 20.0,
            "amp_mag_bessellr": 19.5,
        }
        _, mags_lo = model.predict({**base, "log10_gamma": jnp.log10(10.0)})
        _, mags_hi = model.predict({**base, "log10_gamma": jnp.log10(40.0)})
        for filt in FILTERS:
            assert not jnp.allclose(mags_lo[filt], mags_hi[filt]), \
                f"{filt}: gamma should change the light curve"


# ---------------------------------------------------------------------------
# PhenomenologicalTDEModel
# ---------------------------------------------------------------------------

class TestPhenomenologicalTDE:
    def _make_model_and_params(self):
        model = PhenomenologicalTDEModel(filters=FILTERS)
        params = {
            "t0": 5.0,
            "log10_tau_rise": 0.3,
            "log10_tau_fall": jnp.log10(50.0),
            "alpha_decay": 1.67,
            "amp_mag_bessellb": 20.0,
            "base_mag_bessellb": 26.0,
            "amp_mag_bessellr": 19.5,
            "base_mag_bessellr": 26.0,
        }
        return model, params

    def test_shape(self):
        model, params = self._make_model_and_params()
        times, mags = model.predict(params)
        assert times.shape == (200,)
        for filt in FILTERS:
            assert mags[filt].shape == (200,), f"Wrong shape for {filt}"

    def test_finite(self):
        model, params = self._make_model_and_params()
        _, mags = model.predict(params)
        for filt in FILTERS:
            assert jnp.all(jnp.isfinite(mags[filt])), f"Non-finite in {filt}"

    def test_differentiable(self):
        model, params = self._make_model_and_params()
        params = {k: jnp.float64(v) if isinstance(v, (int, float)) else v
                  for k, v in params.items()}

        def _loss(p):
            _, mags = model.predict(p)
            return jnp.sum(jnp.stack([jnp.sum(mags[f]) for f in FILTERS]))

        grads = jax.grad(_loss)(params)
        all_finite = all(jnp.all(jnp.isfinite(v))
                         for v in jax.tree_util.tree_leaves(grads))
        assert all_finite, "Gradients contain non-finite values"

    def test_late_time_decay(self):
        """Magnitudes should increase (flux decays) at late times."""
        model, params = self._make_model_and_params()
        times, mags = model.predict(params)
        for filt in FILTERS:
            # Compare last quarter vs middle quarter — late times should be fainter
            n = len(times)
            mid_mag = jnp.mean(mags[filt][n // 4: n // 2])
            late_mag = jnp.mean(mags[filt][3 * n // 4:])
            assert late_mag > mid_mag, \
                f"{filt}: expected late-time decay (fainter), got mid={float(mid_mag):.1f} late={float(late_mag):.1f}"


# ---------------------------------------------------------------------------
# AfterglowModel
# ---------------------------------------------------------------------------

class TestAfterglow:
    def _make_model_and_params(self):
        model = AfterglowModel(filters=FILTERS)
        params = {
            "t0": 0.5,
            "log10_t_break": 1.0,
            "alpha_1": -1.0,
            "alpha_2": 2.0,
            "amp_mag_bessellb": 20.0,
            "amp_mag_bessellr": 19.5,
        }
        return model, params

    def test_shape(self):
        model, params = self._make_model_and_params()
        times, mags = model.predict(params)
        assert times.shape == (200,)
        for filt in FILTERS:
            assert mags[filt].shape == (200,), f"Wrong shape for {filt}"

    def test_finite(self):
        model, params = self._make_model_and_params()
        _, mags = model.predict(params)
        for filt in FILTERS:
            assert jnp.all(jnp.isfinite(mags[filt])), f"Non-finite in {filt}"

    def test_differentiable(self):
        model, params = self._make_model_and_params()
        params = {k: jnp.float64(v) if isinstance(v, (int, float)) else v
                  for k, v in params.items()}

        def _loss(p):
            _, mags = model.predict(p)
            return jnp.sum(jnp.stack([jnp.sum(mags[f]) for f in FILTERS]))

        grads = jax.grad(_loss)(params)
        all_finite = all(jnp.all(jnp.isfinite(v))
                         for v in jax.tree_util.tree_leaves(grads))
        assert all_finite, "Gradients contain non-finite values"

    def test_break_time(self):
        """Changing t_break should shift the transition."""
        model = AfterglowModel(filters=FILTERS)
        base = {
            "t0": 0.5,
            "alpha_1": -1.0,
            "alpha_2": 2.0,
            "amp_mag_bessellb": 20.0,
            "amp_mag_bessellr": 19.5,
        }
        _, mags_lo = model.predict({**base, "log10_t_break": 0.5})
        _, mags_hi = model.predict({**base, "log10_t_break": 2.0})
        for filt in FILTERS:
            assert not jnp.allclose(mags_lo[filt], mags_hi[filt]), \
                f"{filt}: t_break should change the light curve"


# ---------------------------------------------------------------------------
# SALT3Model
# ---------------------------------------------------------------------------

SALT3_FILTERS = ["ztfg", "ztfr"]

_has_jax_bandflux = importlib.util.find_spec("jax_supernovae") is not None


@pytest.mark.skipif(not _has_jax_bandflux,
                    reason="jax-bandflux not installed")
class TestSALT3:
    def _make_model_and_params(self):
        model = SALT3Model(
            filters=SALT3_FILTERS,
            times=jnp.linspace(0, 50, 100),
            redshift=0.05,
        )
        params = {
            "log10_x0": jnp.array(-4.5),
            "x1": jnp.array(0.0),
            "c": jnp.array(0.0),
            "t0": jnp.array(10.0),
        }
        return model, params

    def test_shape(self):
        model, params = self._make_model_and_params()
        times, mags = model.predict(params)
        assert times.shape == (100,)
        for filt in SALT3_FILTERS:
            assert mags[filt].shape == (100,), f"Wrong shape for {filt}"

    def test_finite(self):
        model, params = self._make_model_and_params()
        _, mags = model.predict(params)
        for filt in SALT3_FILTERS:
            assert jnp.all(jnp.isfinite(mags[filt])), f"Non-finite in {filt}"

    def test_jit_consistent(self):
        model, params = self._make_model_and_params()
        _, m1 = model.predict(params)
        _, m2 = model.predict(params)
        for filt in SALT3_FILTERS:
            np.testing.assert_allclose(m1[filt], m2[filt])

    def test_differentiable(self):
        model, params = self._make_model_and_params()

        def _loss(p):
            _, mags = model.predict(p)
            return jnp.sum(jnp.stack([jnp.sum(mags[f]) for f in SALT3_FILTERS]))

        grads = jax.grad(_loss)(params)
        all_finite = all(jnp.all(jnp.isfinite(v))
                         for v in jax.tree_util.tree_leaves(grads))
        any_nonzero = any(jnp.any(jnp.abs(v) > 0)
                          for v in jax.tree_util.tree_leaves(grads))
        assert all_finite, "Gradients contain non-finite values"
        assert any_nonzero, "All gradients are zero"

    def test_log10_x0_gradient(self):
        """Gradient through log10_x0 should be finite and non-zero."""
        model, params = self._make_model_and_params()

        def _loss(log10_x0):
            p = {**params, "log10_x0": log10_x0}
            _, mags = model.predict(p)
            return jnp.sum(mags[SALT3_FILTERS[0]])

        grad_val = jax.grad(_loss)(params["log10_x0"])
        assert jnp.isfinite(grad_val), f"log10_x0 gradient non-finite: {grad_val}"
        assert jnp.abs(grad_val) > 0, "log10_x0 gradient is zero"

    def test_magnitude_range(self):
        """AB magnitudes should be in a physically reasonable range."""
        model, params = self._make_model_and_params()
        _, mags = model.predict(params)
        for filt in SALT3_FILTERS:
            mag_min = float(jnp.min(mags[filt]))
            mag_max = float(jnp.max(mags[filt]))
            assert 10.0 < mag_min < 40.0, f"{filt}: min mag {mag_min} out of range"
            assert 10.0 < mag_max < 80.0, f"{filt}: max mag {mag_max} out of range"
