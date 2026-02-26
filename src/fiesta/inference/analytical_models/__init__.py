"""Analytical light-curve models implemented in pure JAX.

Each model is fully JIT-compilable and differentiable so that flowMC's MALA
sampler can compute ``jax.grad`` through the likelihood.  The models follow
the same ``predict()`` contract as the surrogate models:

    (source_frame_times, {filter_name: apparent_mag_array})

This makes them drop-in replacements inside ``CombinedSurrogate`` and
``EMLikelihood``.

All internal physics computations use log10 space to avoid float32 overflow
(e.g. explosion energies ~1e49 erg exceed float32 max ~3.4e38).
"""

from fiesta.inference.analytical_models.base import (
    AnalyticalModel,
    _magnetar_luminosity,
    _compute_diffusion_constants,
    _arnett_diffusion_ode,
    _gauss_legendre_nodes_weights,
    _log10_blackbody_mJy_at_10pc,
)
from fiesta.inference.analytical_models.kilonova_models import (
    MetzgerModel, MetzgerFullModel, OneComponentKilonovaModel,
    MagnetarBoostedKilonovaModel,
)
from fiesta.inference.analytical_models.supernova_models import (
    ArnettModel, NickelCobaltModel, MagnetarPoweredSNModel,
    CSMInteractionModel,
)
from fiesta.inference.analytical_models.shock_powered_models import (
    ShockCoolingModel, ShockedCocoonModel,
)
from fiesta.inference.analytical_models.tde_models import TDEAnalyticalModel
from fiesta.inference.analytical_models.phenomenological_models import (
    PhenomenologicalModel, EvolvingBlackbodyModel,
    BazinModel, VillarModel, PhenomenologicalTDEModel, AfterglowModel,
)

__all__ = [
    "AfterglowModel",
    "AnalyticalModel",
    "ArnettModel",
    "BazinModel",
    "CSMInteractionModel",
    "EvolvingBlackbodyModel",
    "MagnetarBoostedKilonovaModel",
    "MagnetarPoweredSNModel",
    "MetzgerFullModel",
    "MetzgerModel",
    "NickelCobaltModel",
    "OneComponentKilonovaModel",
    "PhenomenologicalModel",
    "PhenomenologicalTDEModel",
    "ShockCoolingModel",
    "ShockedCocoonModel",
    "TDEAnalyticalModel",
    "VillarModel",
]
