"""Export fiesta neural-network surrogates to ONNX format.

Produces self-contained ONNX files with input/output scalers baked into the
graph so that the ONNX model maps directly from raw physics parameters to
absolute magnitudes (LightcurveModel) or log10-flux (FluxModel).

Requires the ``onnx`` optional dependency group::

    pip install fiestaEM[onnx]

Example usage::

    from fiesta.export_onnx import export_lightcurve_model, export_flux_model

    export_lightcurve_model("Bu2025", filters=["besselli", "bessellr"],
                            output_dir="/tmp/onnx_test")
    export_flux_model("Bu2025_MLP", output_dir="/tmp/onnx_test")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError:
    raise ImportError(
        "ONNX export requires the 'onnx' package. "
        "Install it with:  pip install fiestaEM[onnx]"
    )

from fiesta.inference.lightcurve_model import get_default_directory
from fiesta.scalers import (
    DataScaler,
    ImageScaler,
    MinMaxScalerJax,
    PCADecomposer,
    ParameterScaler,
    SVDDecomposer,
    StandardScalerJax,
)

# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------


def _extract_dense_layers(params: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract ``(kernel, bias)`` pairs from a Flax params dict.

    Handles both MLP keys (``layers_0``, ``layers_1``, ...) and
    Decoder keys (same naming).  Returns list in layer order.

    Flax Dense ``kernel`` shape is ``(in_features, out_features)`` which
    matches ONNX ``MatMul([batch, in] @ [in, out])`` directly.
    """
    # Collect layer keys and sort numerically
    layer_keys = sorted(
        (k for k in params if k.startswith("layers_")),
        key=lambda k: int(k.split("_")[1]),
    )
    layers = []
    for key in layer_keys:
        kernel = np.asarray(params[key]["kernel"], dtype=np.float32)
        bias = np.asarray(params[key]["bias"], dtype=np.float32)
        layers.append((kernel, bias))
    return layers


# ---------------------------------------------------------------------------
# Scaler  ->  ONNX node builders
# ---------------------------------------------------------------------------
# Each builder returns ``(nodes, initializers, output_name)``.
# ``nodes`` is a list of ``onnx.NodeProto`` and ``initializers`` a list of
# ``onnx.TensorProto`` constants that must be added to the graph.


def _build_minmax_forward(
    scaler: MinMaxScalerJax, input_name: str, prefix: str
) -> tuple[list, list, str]:
    """``(x - min) / (max - min)``."""
    min_val = np.asarray(scaler.min_val, dtype=np.float32).flatten()
    max_val = np.asarray(scaler.max_val, dtype=np.float32).flatten()
    range_val = max_val - min_val

    init_min = numpy_helper.from_array(min_val, name=f"{prefix}_min")
    init_range = numpy_helper.from_array(range_val, name=f"{prefix}_range")

    sub_out = f"{prefix}_sub"
    div_out = f"{prefix}_out"

    nodes = [
        helper.make_node("Sub", [input_name, init_min.name], [sub_out]),
        helper.make_node("Div", [sub_out, init_range.name], [div_out]),
    ]
    return nodes, [init_min, init_range], div_out


def _build_minmax_inverse(
    scaler: MinMaxScalerJax, input_name: str, prefix: str
) -> tuple[list, list, str]:
    """``x * (max - min) + min``."""
    min_val = np.asarray(scaler.min_val, dtype=np.float32).flatten()
    max_val = np.asarray(scaler.max_val, dtype=np.float32).flatten()
    range_val = max_val - min_val

    init_range = numpy_helper.from_array(range_val, name=f"{prefix}_range")
    init_min = numpy_helper.from_array(min_val, name=f"{prefix}_min")

    mul_out = f"{prefix}_mul"
    add_out = f"{prefix}_out"

    nodes = [
        helper.make_node("Mul", [input_name, init_range.name], [mul_out]),
        helper.make_node("Add", [mul_out, init_min.name], [add_out]),
    ]
    return nodes, [init_range, init_min], add_out


def _build_standard_forward(
    scaler: StandardScalerJax, input_name: str, prefix: str
) -> tuple[list, list, str]:
    """``(x - mu) / sigma``."""
    mu = np.asarray(scaler.mu, dtype=np.float32).flatten()
    sigma = np.asarray(scaler.sigma, dtype=np.float32).flatten()

    init_mu = numpy_helper.from_array(mu, name=f"{prefix}_mu")
    init_sigma = numpy_helper.from_array(sigma, name=f"{prefix}_sigma")

    sub_out = f"{prefix}_sub"
    div_out = f"{prefix}_out"

    nodes = [
        helper.make_node("Sub", [input_name, init_mu.name], [sub_out]),
        helper.make_node("Div", [sub_out, init_sigma.name], [div_out]),
    ]
    return nodes, [init_mu, init_sigma], div_out


def _build_standard_inverse(
    scaler: StandardScalerJax, input_name: str, prefix: str
) -> tuple[list, list, str]:
    """``x * sigma + mu``."""
    mu = np.asarray(scaler.mu, dtype=np.float32).flatten()
    sigma = np.asarray(scaler.sigma, dtype=np.float32).flatten()

    init_sigma = numpy_helper.from_array(sigma, name=f"{prefix}_sigma")
    init_mu = numpy_helper.from_array(mu, name=f"{prefix}_mu")

    mul_out = f"{prefix}_mul"
    add_out = f"{prefix}_out"

    nodes = [
        helper.make_node("Mul", [input_name, init_sigma.name], [mul_out]),
        helper.make_node("Add", [mul_out, init_mu.name], [add_out]),
    ]
    return nodes, [init_sigma, init_mu], add_out


def _build_pca_forward(
    scaler: PCADecomposer, input_name: str, prefix: str
) -> tuple[list, list, str]:
    """``(x - means) @ Vt.T``."""
    means = np.asarray(scaler.means, dtype=np.float32).reshape(1, -1)
    vt_t = np.asarray(scaler.Vt, dtype=np.float32).T  # (n_features, n_components)

    init_means = numpy_helper.from_array(means, name=f"{prefix}_means")
    init_vt_t = numpy_helper.from_array(vt_t, name=f"{prefix}_vt_t")

    sub_out = f"{prefix}_sub"
    mm_out = f"{prefix}_out"

    nodes = [
        helper.make_node("Sub", [input_name, init_means.name], [sub_out]),
        helper.make_node("MatMul", [sub_out, init_vt_t.name], [mm_out]),
    ]
    return nodes, [init_means, init_vt_t], mm_out


def _build_pca_inverse(
    scaler: PCADecomposer, input_name: str, prefix: str
) -> tuple[list, list, str]:
    """``x @ Vt + means``."""
    means = np.asarray(scaler.means, dtype=np.float32).reshape(1, -1)
    vt = np.asarray(scaler.Vt, dtype=np.float32)  # (n_components, n_features)

    init_vt = numpy_helper.from_array(vt, name=f"{prefix}_vt")
    init_means = numpy_helper.from_array(means, name=f"{prefix}_means")

    mm_out = f"{prefix}_mm"
    add_out = f"{prefix}_out"

    nodes = [
        helper.make_node("MatMul", [input_name, init_vt.name], [mm_out]),
        helper.make_node("Add", [mm_out, init_means.name], [add_out]),
    ]
    return nodes, [init_vt, init_means], add_out


def _build_svd_forward(
    scaler: SVDDecomposer, input_name: str, prefix: str
) -> tuple[list, list, str]:
    """MinMax forward then ``x @ VA.T``."""
    nodes, inits, cur = _build_minmax_forward(
        scaler.scaler, input_name, f"{prefix}_mm"
    )
    va_t = np.asarray(scaler.VA, dtype=np.float32).T  # (n_features, svd_ncoeff)
    init_va_t = numpy_helper.from_array(va_t, name=f"{prefix}_va_t")
    mm_out = f"{prefix}_out"
    nodes.append(helper.make_node("MatMul", [cur, init_va_t.name], [mm_out]))
    inits.append(init_va_t)
    return nodes, inits, mm_out


def _build_svd_inverse(
    scaler: SVDDecomposer, input_name: str, prefix: str
) -> tuple[list, list, str]:
    """``x @ VA`` then MinMax inverse."""
    va = np.asarray(scaler.VA, dtype=np.float32)  # (svd_ncoeff, n_features)
    init_va = numpy_helper.from_array(va, name=f"{prefix}_va")
    mm_out = f"{prefix}_mm"
    nodes = [helper.make_node("MatMul", [input_name, init_va.name], [mm_out])]
    inv_nodes, inv_inits, out = _build_minmax_inverse(
        scaler.scaler, mm_out, f"{prefix}_mminv"
    )
    nodes.extend(inv_nodes)
    return nodes, [init_va] + inv_inits, out


# ---------------------------------------------------------------------------
# Dispatcher: scaler -> forward / inverse ONNX nodes
# ---------------------------------------------------------------------------


def _build_scaler_forward(
    scaler, input_name: str, prefix: str
) -> tuple[list, list, str]:
    """Dispatch to the correct forward builder based on scaler type."""
    if isinstance(scaler, ParameterScaler):
        return _build_parameter_scaler_forward(scaler, input_name, prefix)
    if isinstance(scaler, DataScaler):
        return _build_data_scaler_forward(scaler, input_name, prefix)
    if isinstance(scaler, MinMaxScalerJax):
        return _build_minmax_forward(scaler, input_name, prefix)
    if isinstance(scaler, StandardScalerJax):
        return _build_standard_forward(scaler, input_name, prefix)
    if isinstance(scaler, PCADecomposer):
        return _build_pca_forward(scaler, input_name, prefix)
    if isinstance(scaler, SVDDecomposer):
        return _build_svd_forward(scaler, input_name, prefix)
    if isinstance(scaler, ImageScaler):
        raise ValueError(
            "ImageScaler is not supported for ONNX export (requires cubic "
            "resize + edge-fix heuristic)."
        )
    raise TypeError(f"Unsupported scaler type: {type(scaler).__name__}")


def _build_scaler_inverse(
    scaler, input_name: str, prefix: str
) -> tuple[list, list, str]:
    """Dispatch to the correct inverse builder based on scaler type."""
    if isinstance(scaler, DataScaler):
        return _build_data_scaler_inverse(scaler, input_name, prefix)
    if isinstance(scaler, MinMaxScalerJax):
        return _build_minmax_inverse(scaler, input_name, prefix)
    if isinstance(scaler, StandardScalerJax):
        return _build_standard_inverse(scaler, input_name, prefix)
    if isinstance(scaler, PCADecomposer):
        return _build_pca_inverse(scaler, input_name, prefix)
    if isinstance(scaler, SVDDecomposer):
        return _build_svd_inverse(scaler, input_name, prefix)
    if isinstance(scaler, ImageScaler):
        raise ValueError(
            "ImageScaler is not supported for ONNX export (requires cubic "
            "resize + edge-fix heuristic)."
        )
    raise TypeError(f"Unsupported scaler type for inverse: {type(scaler).__name__}")


def _build_parameter_scaler_forward(
    scaler: ParameterScaler, input_name: str, prefix: str
) -> tuple[list, list, str]:
    """Handle ParameterScaler: apply conversion then inner scaler forward.

    Only the ``identity`` conversion is supported.  ``thetaWing_inclination``
    and ``thetCore_inclination`` require computing derived feature columns
    which is not supported in this version.
    """
    from fiesta.scalers import identity as _identity_fn

    if scaler.conversion is not _identity_fn:
        raise ValueError(
            f"ParameterScaler conversion '{scaler.conversion.__name__}' is "
            "not supported for ONNX export. Only the identity conversion is "
            "supported."
        )
    # Identity conversion is a no-op; delegate to inner scaler
    return _build_scaler_forward(scaler.scaler, input_name, f"{prefix}_ps")


def _build_data_scaler_forward(
    scaler: DataScaler, input_name: str, prefix: str
) -> tuple[list, list, str]:
    """Chain forward transforms of all scalers in order."""
    all_nodes: list = []
    all_inits: list = []
    cur = input_name
    for i, s in enumerate(scaler.scalers):
        nodes, inits, cur = _build_scaler_forward(s, cur, f"{prefix}_ds{i}")
        all_nodes.extend(nodes)
        all_inits.extend(inits)
    return all_nodes, all_inits, cur


def _build_data_scaler_inverse(
    scaler: DataScaler, input_name: str, prefix: str
) -> tuple[list, list, str]:
    """Chain inverse transforms of all scalers in reverse order."""
    all_nodes: list = []
    all_inits: list = []
    cur = input_name
    for i, s in enumerate(reversed(scaler.scalers)):
        nodes, inits, cur = _build_scaler_inverse(s, cur, f"{prefix}_dsi{i}")
        all_nodes.extend(nodes)
        all_inits.extend(inits)
    return all_nodes, all_inits, cur


# ---------------------------------------------------------------------------
# MLP  ->  ONNX node builder
# ---------------------------------------------------------------------------


def _build_mlp_nodes(
    layers: list[tuple[np.ndarray, np.ndarray]],
    input_name: str,
    output_name: str,
    prefix: str = "mlp",
) -> tuple[list, list]:
    """Build Dense -> ReLU -> ... -> Dense (no final activation) ONNX nodes.

    Returns ``(nodes, initializers)``.
    """
    nodes: list = []
    inits: list = []
    cur = input_name

    for i, (kernel, bias) in enumerate(layers):
        k_name = f"{prefix}_w{i}"
        b_name = f"{prefix}_b{i}"
        inits.append(numpy_helper.from_array(kernel, name=k_name))
        inits.append(numpy_helper.from_array(bias, name=b_name))

        mm_out = f"{prefix}_mm{i}"
        add_out = f"{prefix}_add{i}"
        nodes.append(helper.make_node("MatMul", [cur, k_name], [mm_out]))
        nodes.append(helper.make_node("Add", [mm_out, b_name], [add_out]))

        is_last = i == len(layers) - 1
        if is_last:
            # Rename last Add output to the desired output_name
            nodes[-1] = helper.make_node("Add", [mm_out, b_name], [output_name])
        else:
            relu_out = f"{prefix}_relu{i}"
            nodes.append(helper.make_node("Relu", [add_out], [relu_out]))
            cur = relu_out

    return nodes, inits


# ---------------------------------------------------------------------------
# CVAE latent zero prepend
# ---------------------------------------------------------------------------


def _build_latent_prepend(
    latent_dim: int, input_name: str, prefix: str = "latent"
) -> tuple[list, list, str]:
    """Prepend ``latent_dim`` zeros to the input along axis=1.

    The input has shape ``[batch, scaled_dim]``; the output has shape
    ``[batch, latent_dim + scaled_dim]``.

    Returns ``(nodes, initializers, output_name)``.
    """
    if latent_dim == 0:
        return [], [], input_name

    nodes: list = []
    inits: list = []

    # Get batch size: Shape -> Gather(dim=0)
    shape_out = f"{prefix}_shape"
    nodes.append(helper.make_node("Shape", [input_name], [shape_out]))

    idx_zero = numpy_helper.from_array(
        np.array(0, dtype=np.int64), name=f"{prefix}_idx0"
    )
    inits.append(idx_zero)
    batch_out = f"{prefix}_batch"
    nodes.append(
        helper.make_node("Gather", [shape_out, idx_zero.name], [batch_out],
                         axis=0)
    )

    # Build shape tensor [batch, latent_dim]
    latent_const = numpy_helper.from_array(
        np.array([latent_dim], dtype=np.int64), name=f"{prefix}_ldim"
    )
    inits.append(latent_const)
    batch_unsq = f"{prefix}_batch_unsq"
    nodes.append(
        helper.make_node("Unsqueeze", [batch_out], [batch_unsq],
                         axes=[0])
    )
    zeros_shape = f"{prefix}_zeros_shape"
    nodes.append(
        helper.make_node("Concat", [batch_unsq, latent_const.name],
                         [zeros_shape], axis=0)
    )

    # ConstantOfShape -> zeros
    zero_val = numpy_helper.from_array(
        np.array([0.0], dtype=np.float32), name=f"{prefix}_zero_val"
    )
    inits.append(zero_val)
    zeros_out = f"{prefix}_zeros"
    nodes.append(
        helper.make_node("ConstantOfShape", [zeros_shape], [zeros_out],
                         value=onnx.helper.make_tensor(
                             "value", TensorProto.FLOAT, [1], [0.0]))
    )

    # Concat [zeros, scaled_input] along axis=1
    concat_out = f"{prefix}_out"
    nodes.append(
        helper.make_node("Concat", [zeros_out, input_name], [concat_out],
                         axis=1)
    )

    return nodes, inits, concat_out


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def _build_onnx_model(
    input_dim: int,
    output_dim: int,
    nodes: list,
    initializers: list,
    opset: int = 17,
) -> onnx.ModelProto:
    """Assemble nodes into a complete ONNX ModelProto with dynamic batch dim."""
    # Input: float32[None, input_dim]
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, input_dim])
    # Output: float32[None, output_dim]
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, output_dim])

    graph = helper.make_graph(
        nodes,
        "fiesta_surrogate",
        [X],
        [Y],
        initializer=initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


# ---------------------------------------------------------------------------
# Metadata JSON
# ---------------------------------------------------------------------------


def _write_metadata(
    path: str,
    model_name: str,
    model_type_str: str,
    architecture: str,
    parameter_names: list[str],
    parameter_ranges: dict,
    times,
    nus,
    onnx_files: dict[str, str],
    input_desc: str,
    output_desc: str,
) -> None:
    meta = {
        "model_name": model_name,
        "model_type": model_type_str,
        "architecture": architecture,
        "parameter_names": parameter_names,
        "parameter_ranges": {
            k: [float(v[0]), float(v[1])] for k, v in parameter_ranges.items()
        },
        "times": [float(t) for t in np.asarray(times)],
        "nus": [float(n) for n in np.asarray(nus)] if nus is not None else None,
        "onnx_files": onnx_files,
        "input_description": input_desc,
        "output_description": output_desc,
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_lightcurve_model(
    name: str,
    filters: list[str] | None = None,
    output_dir: str = ".",
    directory: str | None = None,
) -> list[str]:
    """Export a LightcurveModel to ONNX.  One ``.onnx`` file per filter.

    Parameters
    ----------
    name : str
        Model name, e.g. ``"Bu2025"``.
    filters : list[str] or None
        Filters to export.  ``None`` exports all available filters.
    output_dir : str
        Directory to write ``.onnx`` and metadata JSON files.
    directory : str or None
        Path to the model directory.  ``None`` uses the built-in surrogate.

    Returns
    -------
    list[str]
        Paths to the exported ``.onnx`` files.
    """
    import dill
    import fiesta.train.neuralnets as fiesta_nn

    if directory is None:
        directory = get_default_directory(name)

    # --- Load metadata ---
    metadata_files = [f for f in os.listdir(directory) if f.endswith("_metadata.pkl")]
    if not metadata_files:
        raise FileNotFoundError(f"No metadata file found in {directory}")
    with open(os.path.join(directory, metadata_files[0]), "rb") as f:
        metadata = dill.load(f)

    X_scaler = metadata["X_scaler"]
    y_scaler = metadata["y_scaler"]  # dict[str, scaler]
    parameter_names = metadata["parameter_names"]
    times = metadata["times"]
    from ast import literal_eval
    parameter_distributions = literal_eval(metadata["parameter_distributions"])

    # --- Determine filters ---
    pkl_files = [
        f for f in os.listdir(directory)
        if (f.endswith(".pkl") or f.endswith(".pickle")) and "metadata" not in f
    ]
    available_filters = [f.split(".")[0].split("_", 1)[1] for f in pkl_files]

    if filters is None:
        filters = available_filters
    else:
        missing = [f for f in filters if f not in available_filters]
        if missing:
            raise ValueError(
                f"Filters {missing} not found. Available: {available_filters}"
            )

    os.makedirs(output_dir, exist_ok=True)

    # --- Figure out input_dim from the X_scaler ---
    input_dim = len(parameter_names)

    exported_files = []
    onnx_file_map = {}

    for filt in filters:
        # Load MLP for this filter
        filt_pkl = [f for f in pkl_files if f.split(".")[0].split("_", 1)[1] == filt]
        if not filt_pkl:
            continue
        state, config = fiesta_nn.MLP.load_model(
            os.path.join(directory, filt_pkl[0])
        )
        layers = _extract_dense_layers(dict(state.params))
        output_dim = layers[-1][1].shape[0]  # last bias size

        # Build graph: X_scaler forward -> MLP -> y_scaler[filt] inverse
        all_nodes: list = []
        all_inits: list = []

        # Input scaler
        n, ini, cur = _build_scaler_forward(X_scaler, "input", "xscaler")
        all_nodes.extend(n)
        all_inits.extend(ini)

        # MLP
        mlp_n, mlp_ini = _build_mlp_nodes(layers, cur, "mlp_out", "mlp")
        all_nodes.extend(mlp_n)
        all_inits.extend(mlp_ini)

        # Output inverse scaler
        n, ini, cur = _build_scaler_inverse(y_scaler[filt], "mlp_out", "yscaler")
        all_nodes.extend(n)
        all_inits.extend(ini)

        # Rename final output
        all_nodes.append(helper.make_node("Identity", [cur], ["output"]))

        model = _build_onnx_model(input_dim, output_dim, all_nodes, all_inits)

        fname = f"{name}_{filt}.onnx"
        out_path = os.path.join(output_dir, fname)
        onnx.save(model, out_path)
        exported_files.append(out_path)
        onnx_file_map[filt] = fname

    # Write metadata JSON
    _write_metadata(
        os.path.join(output_dir, f"{name}_metadata.json"),
        model_name=name,
        model_type_str="LightcurveModel",
        architecture="MLP",
        parameter_names=parameter_names,
        parameter_ranges=parameter_distributions,
        times=times,
        nus=None,
        onnx_files=onnx_file_map,
        input_desc="Raw physics parameters",
        output_desc="Absolute magnitudes at model time grid",
    )

    return exported_files


def export_flux_model(
    name: str,
    output_dir: str = ".",
    directory: str | None = None,
) -> str:
    """Export a FluxModel to ONNX.  One ``.onnx`` file.

    Parameters
    ----------
    name : str
        Model name, e.g. ``"Bu2025_MLP"`` or ``"pbag_gaussian_CVAE"``.
    output_dir : str
        Directory to write ``.onnx`` and metadata JSON files.
    directory : str or None
        Path to the model directory.  ``None`` uses the built-in surrogate.

    Returns
    -------
    str
        Path to the exported ``.onnx`` file.
    """
    import dill
    import jax.numpy as jnp
    import fiesta.train.neuralnets as fiesta_nn

    if directory is None:
        directory = get_default_directory(name)

    # --- Load metadata ---
    metadata_files = [f for f in os.listdir(directory) if f.endswith("_metadata.pkl")]
    if not metadata_files:
        raise FileNotFoundError(f"No metadata file found in {directory}")
    with open(os.path.join(directory, metadata_files[0]), "rb") as f:
        metadata = dill.load(f)

    X_scaler = metadata["X_scaler"]
    y_scaler = metadata["y_scaler"]
    model_type = metadata["model_type"]
    parameter_names = metadata["parameter_names"]
    times = metadata["times"]
    nus = metadata.get("nus", None)
    from ast import literal_eval
    parameter_distributions = literal_eval(metadata["parameter_distributions"])

    # --- Load network ---
    nn_files = [
        f for f in os.listdir(directory)
        if (f.endswith(".pkl") or f.endswith(".pickle")) and "metadata" not in f
    ]
    nn_file = os.path.join(directory, nn_files[0])

    if model_type == "MLP":
        state, config = fiesta_nn.MLP.load_model(nn_file)
        latent_dim = 0
    elif model_type == "CVAE":
        state, config = fiesta_nn.CVAE.load_model(nn_file)  # decoder only
        # Determine latent dim the same way FluxModel does
        x_tilde_dim = X_scaler.transform(
            jnp.zeros(len(parameter_names)).reshape(1, -1)
        ).shape[1]
        first_layer_in = state.params["layers_0"]["kernel"].shape[0]
        latent_dim = first_layer_in - x_tilde_dim
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    layers = _extract_dense_layers(dict(state.params))
    input_dim = len(parameter_names)
    output_dim = layers[-1][1].shape[0]

    os.makedirs(output_dir, exist_ok=True)

    # Build graph: X_scaler forward -> [latent prepend] -> MLP -> y_scaler inverse
    all_nodes: list = []
    all_inits: list = []

    # Input scaler
    n, ini, cur = _build_scaler_forward(X_scaler, "input", "xscaler")
    all_nodes.extend(n)
    all_inits.extend(ini)

    # Latent zero prepend (for CVAE)
    n, ini, cur = _build_latent_prepend(latent_dim, cur, "latent")
    all_nodes.extend(n)
    all_inits.extend(ini)

    # MLP / Decoder
    mlp_n, mlp_ini = _build_mlp_nodes(layers, cur, "mlp_out", "mlp")
    all_nodes.extend(mlp_n)
    all_inits.extend(mlp_ini)

    # Output inverse scaler
    n, ini, cur = _build_scaler_inverse(y_scaler, "mlp_out", "yscaler")
    all_nodes.extend(n)
    all_inits.extend(ini)

    # Rename final output
    all_nodes.append(helper.make_node("Identity", [cur], ["output"]))

    model = _build_onnx_model(input_dim, output_dim, all_nodes, all_inits)

    fname = f"{name}.onnx"
    out_path = os.path.join(output_dir, fname)
    onnx.save(model, out_path)

    arch = "CVAE_decoder" if model_type == "CVAE" else "MLP"
    _write_metadata(
        os.path.join(output_dir, f"{name}_metadata.json"),
        model_name=name,
        model_type_str="FluxModel",
        architecture=arch,
        parameter_names=parameter_names,
        parameter_ranges=parameter_distributions,
        times=times,
        nus=nus,
        onnx_files={name: fname},
        input_desc="Raw physics parameters",
        output_desc="log10 flux density (flat array, n_nus * n_times)",
    )

    return out_path
