# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`fiesta` (fiestaEM on PyPI) is a JAX-based package for fast Bayesian inference of kilonova (KN) and gamma-ray burst (GRB) afterglow lightcurves, but also other astrophysical transients. 
It mainly uses pre-trained neural-network surrogate models in place of expensive radiative-transfer/hydrodynamics simulations, but also has some analytical models available.

## Commands

Install in editable mode (required for development):
```bash
pip install -e .
# with optional samplers (flowMC, blackjax, numpyro):
pip install -e ".[samplers]"
# with GRB analytical model support (afterglowpy):
pip install -e ".[grb]"
```

Run the full test suite with coverage (mirrors CI):
```bash
python -m coverage run --source fiesta -m pytest tests/*.py
```

Run a single test file or test:
```bash
python -m pytest tests/test_models.py
python -m pytest tests/test_models.py::test_CVAE_surrogates
```

Lint (CI runs this per-Python-version, threshold `--fail-under=5`, only checking errors/fatals since `C,R,W` are disabled):
```bash
pylint --fail-under=5 --disable=C,R,W $(git ls-files 'src/*.py')
```

Build docs locally:
```bash
pip install -e ".[docs]"
sphinx-build docs docs/_build/html
# live-reload:
sphinx-autobuild docs docs/_build/html
# check for broken refs (matches CI):
sphinx-build -W --keep-going docs docs/_build/html
```

Fetch built-in surrogates (models attempt to load automatically when first called, but useful for an overview):
```bash
python -c "from fiesta.surrogates import download_recommended_surrogates; download_recommended_surrogates()"
python -c "from fiesta.surrogates import print_built_in_surrogates; print_built_in_surrogates()"
python -c "from fiesta.surrogates import print_downloadable_surrogates; print_downloadable_surrogates()"
```

Note: by default only CPU JAX is installed. GPU support requires `pip install fiestaEM[gpu]` or manually installing `jax[cuda13]`. GPU is strongly recommended for training and for full inference runs; CPU is fine for quick tests/examples.

## Architecture

### Three independent pieces: training, models, and inference

The package has a training pipeline (`src/fiesta/train/`) that produces surrogate model artifacts, a model layer (`src/fiesta/models/`) that defines everything with a `predict()` lightcurve interface (trained surrogates and analytical physics models alike), and an inference pipeline (`src/fiesta/inference/`) that consumes model objects to run Bayesian parameter estimation. Training and models interact only through the on-disk surrogate format (a `.pkl` metadata file + Flax/dill model files in a `model/` subdirectory).

### Surrogate model loading (`src/fiesta/surrogates/`)

- `manage_surrogates.py` implements three ways surrogates get resolved: (1) **built-in** — surrogates present under `src/fiesta/surrogates/{KN,GRB}/<model_name>/`, discovered by directory scan (`built_in_surrogates()`); (2) **download** from the `nuclear-multimessenger-astronomy/fiesta-surrogates` Hugging Face repo (`download_surrogate()`), which lands them in a built-in location so they become auto-loadable; (3) **explicit directory** — any `.pkl`/model directory path passed by the user.
- `models/surrogate_models.py`'s `get_default_directory()` is the resolution order used when a model is loaded by name only: check built-ins first, then attempt a Hugging Face download, else raise.
- Surrogate artifacts are large (some GB); mock/reduced training data lives in `examples/training/data/`, and full training data (>10GB) is not distributed in-repo — contact the maintainers for it (see README).

### Model layer (`src/fiesta/models/`)

- **`surrogate_models.py`** — `SurrogateModel` (abstract base) → `FluxSurrogate` / `LightcurveSurrogate` subclasses that load a trained surrogate and predict lightcurves/spectra from physical parameters, plus `CombinedSurrogate` for combining several models (surrogate or analytical) into one joint-emission predictor.
- **`analytical_models/`** — non-surrogate, ab-initio physical models (kilonova, phenomenological, SALT3, shock-powered, supernova, TDE) sharing an `AnalyticalModel` base (`base.py`), useful for validation against or in place of surrogates. All models across both submodules share the same `predict(x) -> (times, {filter: mag})` contract, though the two hierarchies aren't yet unified under one shared base class (`AnalyticalModel` is currently missing a `.name` attribute that `SurrogateModel` has — a known gap, see `CombinedSurrogate.__repr__`).
- `fiesta.inference.tables` (a data-only, `__init__.py`-less namespace package under `src/fiesta/inference/tables/`) holds `csm_table.txt`, referenced by `analytical_models/supernova_models.py`'s CSM interaction model — it did not move with the rest of the model code.

### Inference pipeline (`src/fiesta/inference/`)

Central orchestrator is `Fiesta` in `fiesta.py`, which wires together:
- **`prior/`** — `Prior`/`prior_dict.py` define parameter priors; sampling and naming conventions here must match `parameter_names` on the model being used.
- **`likelihood.py`** — `EMLikelihood` combines observed photometry (detections + non-detections, per-filter times) with a model and systematic-error setup to produce a log-likelihood usable by any sampler.
- **`systematic.py`** — sets up systematic uncertainty either as a single fixed `error_budget` (mag) or from a YAML config (`setup_systematic_from_file`, see `examples/inference/systematics_file_*.yaml` for the schema).
- **`samplers/`** — pluggable backends behind a common interface: `flowmc` (default, optional dep `flowMC`), `blackjax-smc`, `numpyro-svi`, `blackjax_nested_sampling`. `Fiesta.__init__` lazily imports the chosen backend so uninstalled optional samplers don't break unrelated code paths.
- **`injection.py`** — synthetic-data injection for testing recovery of known parameters.
- **`plot.py`** — corner plots and lightcurve plotting utilities (`corner_plot`, `LightcurvePlotter`); plotting checks for LaTeX availability before using it.
- **`wrappers.py`** — convenience wrappers, e.g. exposing fiesta models under other inference frameworks' APIs (see `examples/inference/inference_KN_jesterAPI.py`).

`Fiesta.__init__` validates that the observed data time range is physically reachable given the surrogate's source-frame time grid and the prior's redshift range (raises `ValueError` early rather than failing deep in the sampler) — keep this check in mind if you touch time-range/redshift handling.

### Training pipeline (`src/fiesta/train/`)

- **`DataManager.py`** — loads and preprocesses training data from `.h5` files, handles grid interpolation/masking (`array_mask_from_interval`) and redshift transforms; shared by both trainer types.
- **`FluxTrainer.py`** — trains a surrogate that predicts a full spectral flux density array (wavelength/frequency-resolved).
- **`LightcurveTrainer.py`** — trains a per-filter collection of surrogates predicting lightcurves directly in specific photometric filters.
- **`neuralnets.py` / `nn_architectures.py`** — Flax model definitions (MLP and CVAE architectures) and training-state helpers used by both trainers.
- **`Benchmarker.py`** — evaluates a trained surrogate's accuracy against held-out/validation data.
- Trainers write out the `metadata.pkl` (scalers, parameter names/distributions, time/frequency grids, model type) + model weights that `SurrogateModel.load_metadata()` (in `fiesta.models.surrogate_models`) expects at load time — the two must stay in sync if you change either side.
- Example training scripts: `examples/training/training_MLP.py` (LightcurveTrainer/MLP) and `examples/training/training_CVAE.py` (FluxTrainer/CVAE), using reduced mock data in `examples/training/data/`.

### Supporting modules

- **`conversions.py`** — magnitude/flux conversions, redshift application (`apply_redshift`), apparent/absolute magnitude conversions — used by both training and inference.
- **`scalers.py`** — `MinMaxScalerJax` and related parameter/data scalers used to normalize inputs/outputs for the neural nets.
- **`filters.py`** — `Filter` class and photometric filter/bandpass definitions (uses `sncosmo`).
- **`extinction.py`** — dust extinction corrections.
- **`constants.py`** — shared physical constants.
- **`logging.py`** — project-wide `logger` used everywhere instead of bare `print`.

### Containers and CI

- `containers/dockerfile-OSG` and `dockerfile-OSG-gpu` build CPU/GPU runtime images for OSG (Open Science Grid) use; CI (`.github/workflows/docker-build.yml`) only validates that these build, it never pushes.
- `.github/workflows/unittest.yml` runs the pytest suite across Python 3.11–3.14 via conda/mamba.
- `.github/workflows/pylint.yml` runs pylint (errors/fatals only) across the same Python versions.
- `.github/workflows/docs.yml` builds and presumably publishes the Sphinx docs.
