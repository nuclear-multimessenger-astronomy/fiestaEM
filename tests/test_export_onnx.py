"""Tests for ONNX export of fiesta surrogates."""

import json
import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from fiesta.export_onnx import export_flux_model, export_lightcurve_model

# Use built-in surrogates shipped in src/fiesta/surrogates/
LC_MODEL = "Bu2025"
LC_FILTERS = ["besselli", "bessellr"]

# Bu2025_MLP is a FluxModel with identity conversion (no thetaWing)
FLUX_MODEL_MLP = "Bu2025_MLP"


@pytest.fixture(scope="module")
def lc_export_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        export_lightcurve_model(LC_MODEL, filters=LC_FILTERS, output_dir=tmpdir)
        yield tmpdir


@pytest.fixture(scope="module")
def flux_mlp_export_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        export_flux_model(FLUX_MODEL_MLP, output_dir=tmpdir)
        yield tmpdir


# ── LightcurveModel tests ──────────────────────────────────────────────


def test_export_lc_model_creates_files(lc_export_dir):
    """Exported ONNX files and metadata JSON exist for each filter."""
    for filt in LC_FILTERS:
        onnx_path = os.path.join(lc_export_dir, f"{LC_MODEL}_{filt}.onnx")
        assert os.path.isfile(onnx_path), f"Missing {onnx_path}"
    json_path = os.path.join(lc_export_dir, f"{LC_MODEL}_metadata.json")
    assert os.path.isfile(json_path), f"Missing {json_path}"


def test_export_lc_model_onnx_matches_jax(lc_export_dir):
    """ONNX Runtime output matches JAX predict for LightcurveModel."""
    from fiesta.inference.lightcurve_model import LightcurveModel

    jax_model = LightcurveModel(LC_MODEL, filters=LC_FILTERS)

    # Build a sample input inside the parameter ranges
    param_names = jax_model.parameter_names
    param_ranges = jax_model.parameter_distributions
    sample = {}
    for name in param_names:
        lo, hi = param_ranges[name][:2]
        sample[name] = (float(lo) + float(hi)) / 2.0

    # JAX prediction (absolute mag: set dist=1e-5, z=0)
    sample_abs = dict(sample, luminosity_distance=1e-5, redshift=0.0)
    _, jax_mags = jax_model.predict(sample_abs)

    # ONNX prediction — use sample_abs so redshift/distance match JAX
    x_array = np.array(
        [[sample_abs[n] for n in param_names]], dtype=np.float32
    )
    for filt in LC_FILTERS:
        onnx_path = os.path.join(lc_export_dir, f"{LC_MODEL}_{filt}.onnx")
        sess = ort.InferenceSession(onnx_path)
        onnx_out = sess.run(None, {"input": x_array})[0]
        jax_out = np.asarray(jax_mags[filt])
        np.testing.assert_allclose(
            onnx_out.flatten(), jax_out.flatten(), rtol=1e-4, atol=1e-4,
            err_msg=f"Mismatch for filter {filt}",
        )


# ── FluxModel MLP tests ───────────────────────────────────────────────


def test_export_flux_model_mlp_creates_files(flux_mlp_export_dir):
    """Exported ONNX file and metadata JSON exist for FluxModel MLP."""
    onnx_path = os.path.join(flux_mlp_export_dir, f"{FLUX_MODEL_MLP}.onnx")
    assert os.path.isfile(onnx_path), f"Missing {onnx_path}"
    json_path = os.path.join(flux_mlp_export_dir, f"{FLUX_MODEL_MLP}_metadata.json")
    assert os.path.isfile(json_path), f"Missing {json_path}"


def test_export_flux_model_mlp_matches_jax(flux_mlp_export_dir):
    """ONNX Runtime output matches JAX for FluxModel MLP."""
    from fiesta.inference.lightcurve_model import FluxModel

    jax_model = FluxModel(FLUX_MODEL_MLP, filters=["besselli"])

    param_names = jax_model.parameter_names
    param_ranges = jax_model.parameter_distributions
    sample = {}
    for name in param_names:
        lo, hi = param_ranges[name][:2]
        sample[name] = (float(lo) + float(hi)) / 2.0

    # JAX: replicate the forward pipeline (scaler -> MLP -> inverse scaler)
    x_array = jnp.array([sample[n] for n in param_names]).reshape(1, -1)
    x_tilde = jax_model.X_scaler.transform(x_array)
    x_tilde_flat = x_tilde.reshape(-1)
    x_tilde_flat = jnp.concatenate((jax_model.latent_vector, x_tilde_flat))
    y_tilde = jax_model.models.apply_fn(
        {"params": jax_model.models.params}, x_tilde_flat
    )
    jax_out = np.asarray(jax_model.y_scaler.inverse_transform(y_tilde))

    # ONNX
    onnx_path = os.path.join(flux_mlp_export_dir, f"{FLUX_MODEL_MLP}.onnx")
    sess = ort.InferenceSession(onnx_path)
    x_np = np.array([[sample[n] for n in param_names]], dtype=np.float32)
    onnx_out = sess.run(None, {"input": x_np})[0]

    np.testing.assert_allclose(
        onnx_out.flatten(), jax_out.flatten(), rtol=1e-4, atol=1e-4,
    )


# ── Dynamic batch ─────────────────────────────────────────────────────


def test_dynamic_batch(lc_export_dir):
    """ONNX model handles batch=1 and batch=4."""
    from fiesta.inference.lightcurve_model import LightcurveModel

    jax_model = LightcurveModel(LC_MODEL, filters=LC_FILTERS[:1])
    n_params = len(jax_model.parameter_names)

    filt = LC_FILTERS[0]
    onnx_path = os.path.join(lc_export_dir, f"{LC_MODEL}_{filt}.onnx")
    sess = ort.InferenceSession(onnx_path)

    for batch_size in [1, 4]:
        x = np.random.default_rng(42).standard_normal(
            (batch_size, n_params)
        ).astype(np.float32)
        out = sess.run(None, {"input": x})[0]
        assert out.shape[0] == batch_size, (
            f"Expected batch dim {batch_size}, got {out.shape[0]}"
        )
        assert out.ndim == 2


# ── Metadata JSON ─────────────────────────────────────────────────────


def test_metadata_json_content(lc_export_dir):
    """Metadata JSON has all expected keys and matches model."""
    from fiesta.inference.lightcurve_model import LightcurveModel

    jax_model = LightcurveModel(LC_MODEL, filters=LC_FILTERS)

    json_path = os.path.join(lc_export_dir, f"{LC_MODEL}_metadata.json")
    with open(json_path) as f:
        meta = json.load(f)

    expected_keys = {
        "model_name", "model_type", "architecture", "parameter_names",
        "parameter_ranges", "times", "nus", "onnx_files",
        "input_description", "output_description",
    }
    assert expected_keys.issubset(set(meta.keys()))
    assert meta["model_name"] == LC_MODEL
    assert meta["model_type"] == "LightcurveModel"
    assert meta["parameter_names"] == jax_model.parameter_names
    assert set(meta["onnx_files"].keys()) == set(LC_FILTERS)


# ── ONNX checker ──────────────────────────────────────────────────────


def test_onnx_checker_passes(lc_export_dir, flux_mlp_export_dir):
    """All exported models pass onnx.checker.check_model."""
    for filt in LC_FILTERS:
        path = os.path.join(lc_export_dir, f"{LC_MODEL}_{filt}.onnx")
        model = onnx.load(path)
        onnx.checker.check_model(model)

    path = os.path.join(flux_mlp_export_dir, f"{FLUX_MODEL_MLP}.onnx")
    model = onnx.load(path)
    onnx.checker.check_model(model)
