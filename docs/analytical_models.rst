Analytical Models
-----------------

Fiesta includes a library of analytical (physics-based) light-curve models
implemented in pure JAX.  Unlike the surrogate models, which are neural-network
approximations trained on simulation data, analytical models evaluate
closed-form or ODE-based physics equations directly.

Key properties:

* **JIT-compilable and differentiable** -- compatible with flowMC's MALA
  gradient sampler via ``jax.grad``.
* **Drop-in replacements** for surrogates -- every analytical model exposes the
  same ``predict()`` interface and can be used inside ``CombinedSurrogate`` and
  ``EMLikelihood`` without any code changes.
* **Float32-safe** -- all internal computations use log10 space to avoid
  overflow (e.g. explosion energies ~1e49 erg exceed float32 max ~3.4e38).

When to use analytical models vs surrogates:

* Use **surrogates** when a fast, pre-trained approximation of a complex
  simulation (e.g. radiative-transfer kilonova or jet afterglow) is available
  and sufficient.
* Use **analytical models** when you need direct physics interpretability,
  want to combine a surrogate with a simple analytical component (e.g.
  afterglow surrogate + kilonova analytical), or when no surrogate exists for
  the transient class of interest.


Usage
^^^^^

**Instantiating a model**

Every analytical model takes a list of filter names and an array of
source-frame times (days):

.. code:: python

    import jax.numpy as jnp
    from fiesta.inference.analytical_models import OneComponentKilonovaModel

    model = OneComponentKilonovaModel(
        filters=["bessellb", "bessellv", "bessellr"],
        times=jnp.geomspace(0.1, 30.0, 100),
    )

**Calling predict**

The ``predict`` method accepts a dictionary of parameter values and returns
``(source_frame_times, {filter_name: apparent_mag_array})``:

.. code:: python

    params = {
        "log10_mej": -1.5,
        "log10_vej": -0.5,
        "log10_kappa": -0.3,
        "luminosity_distance": 100.0,   # Mpc
        "redshift": 0.022,
    }

    t_days, mag_app = model.predict(params)
    # mag_app["bessellb"] is an array of apparent magnitudes


**Combining with a surrogate**

Analytical models are interchangeable with surrogates inside
``CombinedSurrogate``.  For example, to model a kilonova + GRB afterglow
where the afterglow is a trained surrogate:

.. code:: python

    from fiesta.inference.lightcurve_model import AfterglowFlux, CombinedSurrogate
    from fiesta.inference.analytical_models import OneComponentKilonovaModel

    afterglow = AfterglowFlux(
        name="afgpy_gaussian_CVAE",
        filters=["bessellb", "bessellv", "bessellr"],
    )

    kilonova = OneComponentKilonovaModel(
        filters=["bessellb", "bessellv", "bessellr"],
        times=jnp.geomspace(0.1, 30.0, 100),
    )

    combined = CombinedSurrogate(
        models=[afterglow, kilonova],
        sample_times=jnp.geomspace(0.1, 300.0, 200),
    )

    t, mags = combined.predict(params)

The fluxes of both models are added in each photometric band.  If a model
falls outside the time range of the ``sample_times`` it is set to
``jnp.inf`` mag there.


Model Catalogue
^^^^^^^^^^^^^^^

Phenomenological Models
"""""""""""""""""""""""

Empirical shape functions that parameterize light-curve morphology directly
in magnitudes, without underlying radiation physics.

**BazinModel**

Exponential rise and fall (Bazin et al. 2009).  Useful as a generic empirical
template for any transient.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Units
   * - ``t0``
     - Explosion / reference time
     - days
   * - ``log10_tau_rise``
     - Rise timescale (log10)
     - log10(days)
   * - ``log10_tau_fall``
     - Fall timescale (log10)
     - log10(days)
   * - ``amp_mag_{f}``
     - Peak amplitude per filter *f*
     - mag
   * - ``base_mag_{f}``
     - Baseline flux per filter *f*
     - mag

**VillarModel**

Piecewise linear rise + power-law decay with a smooth sigmoid transition
(Villar et al. 2017).

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Units
   * - ``t0``
     - Explosion / reference time
     - days
   * - ``log10_tau_rise``
     - Rise timescale (log10)
     - log10(days)
   * - ``log10_tau_fall``
     - Fall timescale (log10)
     - log10(days)
   * - ``beta_slope``
     - Power-law decay slope before peak
     - --
   * - ``log10_gamma``
     - Transition point (log10)
     - log10(days)
   * - ``amp_mag_{f}``
     - Amplitude per filter *f*
     - mag

**AfterglowModel**

Smooth broken power-law for GRB/jet afterglow light curves.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Units
   * - ``t0``
     - Reference time
     - days
   * - ``log10_t_break``
     - Break time (log10)
     - log10(days)
   * - ``alpha_1``
     - Pre-break power-law index
     - --
   * - ``alpha_2``
     - Post-break power-law index
     - --
   * - ``amp_mag_{f}``
     - Amplitude per filter *f*
     - mag

**PhenomenologicalTDEModel**

Sigmoid rise + power-law decay for tidal disruption events.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Units
   * - ``t0``
     - Explosion / reference time
     - days
   * - ``log10_tau_rise``
     - Rise timescale (log10)
     - log10(days)
   * - ``log10_tau_fall``
     - Fall timescale (log10)
     - log10(days)
   * - ``alpha_decay``
     - Power-law decay index
     - --
   * - ``amp_mag_{f}``
     - Amplitude per filter *f*
     - mag
   * - ``base_mag_{f}``
     - Baseline flux per filter *f*
     - mag

**EvolvingBlackbodyModel**

Empirical piecewise power-law evolution of temperature and radius, converted
to broadband magnitudes via blackbody SED.  Unlike the other phenomenological
models this computes bolometric luminosity from ``L = 4 pi R^2 sigma T^4``.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Units
   * - ``log10_temperature_0``
     - Initial temperature (log10)
     - log10(K)
   * - ``log10_radius_0``
     - Initial radius (log10)
     - log10(cm)
   * - ``temp_rise_index``
     - Temperature power-law index before peak
     - --
   * - ``temp_decline_index``
     - Temperature power-law index after peak
     - --
   * - ``temp_peak_time``
     - Time of temperature maximum
     - days
   * - ``radius_rise_index``
     - Radius power-law index before peak
     - --
   * - ``radius_decline_index``
     - Radius power-law index after peak
     - --
   * - ``radius_peak_time``
     - Time of radius maximum
     - days

Supernova Models
""""""""""""""""

Models for core-collapse and thermonuclear supernovae powered by
radioactive decay, magnetar spin-down, or circumstellar interaction.

**ArnettModel**

Nickel-56 / Cobalt-56 radioactive decay without diffusion (Arnett 1982).
Uses Gauss-Legendre quadrature for the decay integral.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Units
   * - ``tau_m``
     - Diffusion timescale
     - days
   * - ``log10_mni``
     - Nickel-56 mass (log10)
     - log10(Msun)
   * - ``v_phot``
     - Photospheric velocity
     - 1e9 cm/s

**NickelCobaltModel**

Nickel-56 / Cobalt-56 decay with Arnett (1982) diffusion integral.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Units
   * - ``f_nickel``
     - Fraction of ejecta in Nickel-56
     - --
   * - ``log10_mej``
     - Ejecta mass (log10)
     - log10(Msun)
   * - ``log10_vej``
     - Ejecta velocity (log10)
     - log10(km/s)
   * - ``log10_kappa``
     - Optical opacity (log10)
     - log10(cm^2/g)
   * - ``log10_kappa_gamma``
     - Gamma-ray opacity (log10)
     - log10(cm^2/g)

**MagnetarPoweredSNModel**

Magnetar spin-down engine with Arnett diffusion.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Units
   * - ``log10_p0``
     - Initial spin period (log10)
     - log10(ms)
   * - ``log10_bp``
     - Polar magnetic field (log10, in units of 1e14 G)
     - log10(1e14 G)
   * - ``mass_ns``
     - Neutron star mass
     - Msun
   * - ``theta_pb``
     - Angle between spin and B-field axes
     - radians
   * - ``log10_mej``
     - Ejecta mass (log10)
     - log10(Msun)
   * - ``log10_vej``
     - Ejecta velocity (log10)
     - log10(km/s)
   * - ``log10_kappa``
     - Optical opacity (log10)
     - log10(cm^2/g)
   * - ``log10_kappa_gamma``
     - Gamma-ray opacity (log10)
     - log10(cm^2/g)

**CSMInteractionModel**

Circumstellar medium shock interaction following the Chevalier (1982)
self-similar solution for forward and reverse shocks.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Units
   * - ``log10_mej``
     - Ejecta mass (log10)
     - log10(Msun)
   * - ``log10_csm_mass``
     - CSM mass (log10)
     - log10(Msun)
   * - ``log10_vej``
     - Ejecta velocity (log10)
     - log10(km/s)
   * - ``eta``
     - CSM density profile exponent
     - --
   * - ``log10_rho``
     - CSM density amplitude (log10)
     - log10(g/cm^{eta+3})
   * - ``log10_kappa``
     - Opacity (log10)
     - log10(cm^2/g)
   * - ``log10_r0``
     - CSM inner radius (log10)
     - log10(AU)

Kilonova Models
"""""""""""""""

Models for kilonovae (neutron-star merger ejecta) powered by r-process
radioactive heating.

**MetzgerModel**

Multi-shell ODE kilonova with 300 mass shells (Metzger 2017).  Includes
early neutron precursor and late r-process heating.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Units
   * - ``log10_mej``
     - Ejecta mass (log10)
     - log10(Msun)
   * - ``log10_vej``
     - Minimum ejecta velocity (log10)
     - log10(c)
   * - ``beta``
     - Velocity power-law index
     - --
   * - ``log10_kappa_r``
     - R-process opacity (log10)
     - log10(cm^2/g)

**MetzgerFullModel**

Multi-shell ODE kilonova with 200 shells using linear velocity spacing and
Barnes+16 thermalisation efficiency.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Units
   * - ``log10_mej``
     - Ejecta mass (log10)
     - log10(Msun)
   * - ``log10_vej``
     - Minimum ejecta velocity (log10)
     - log10(c)
   * - ``beta``
     - Velocity power-law index
     - --
   * - ``log10_kappa_r``
     - R-process opacity (log10)
     - log10(cm^2/g)

Constructor options: ``neutron_precursor`` (bool, default True),
``vmax`` (float, default 0.7).

**OneComponentKilonovaModel**

Single-component kilonova using a diffusion integral (no ODE).

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Units
   * - ``log10_mej``
     - Ejecta mass (log10)
     - log10(Msun)
   * - ``log10_vej``
     - Ejecta velocity (log10)
     - log10(c)
   * - ``log10_kappa``
     - Gray opacity (log10)
     - log10(cm^2/g)

Constructor options: ``temperature_floor`` (float, e.g. 4000.0 K).

**MagnetarBoostedKilonovaModel**

Multi-shell ODE kilonova (200 shells) with additional magnetar spin-down
heating injected into the innermost shell.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Units
   * - ``log10_mej``
     - Ejecta mass (log10)
     - log10(Msun)
   * - ``log10_vej``
     - Minimum ejecta velocity (log10)
     - log10(c)
   * - ``beta``
     - Velocity power-law index
     - --
   * - ``log10_kappa_r``
     - R-process opacity (log10)
     - log10(cm^2/g)
   * - ``log10_p0``
     - Initial spin period (log10)
     - log10(ms)
   * - ``log10_bp``
     - Polar magnetic field (log10, in units of 1e14 G)
     - log10(1e14 G)
   * - ``mass_ns``
     - Neutron star mass
     - Msun
   * - ``theta_pb``
     - Angle between spin and B-field axes
     - radians
   * - ``thermalisation_efficiency``
     - Magnetar thermalisation fraction
     - --

Constructor options: ``neutron_precursor`` (bool), ``pair_cascade`` (bool),
``vmax`` (float).

Shock-Powered Models
""""""""""""""""""""

Models for early emission powered by shock breakout and shock cooling.

**ShockCoolingModel**

Shock-cooling emission following Piro (2021) with analytic piecewise
solution.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Units
   * - ``log10_Menv``
     - Envelope mass (log10)
     - log10(Msun)
   * - ``log10_Renv``
     - Envelope radius (log10)
     - log10(Rsun)
   * - ``log10_Ee``
     - Explosion energy (log10)
     - log10(erg)

**ShockedCocoonModel**

Jet cocoon shock cooling with a fully algebraic (closed-form) solution.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Units
   * - ``log10_mej``
     - Ejecta mass (log10)
     - log10(Msun)
   * - ``log10_vej``
     - Ejecta velocity (log10)
     - log10(c)
   * - ``eta``
     - Density profile exponent
     - --
   * - ``log10_tshock``
     - Shock breakout time (log10)
     - log10(s)
   * - ``shocked_fraction``
     - Fraction of ejecta shocked
     - --
   * - ``cos_theta_cocoon``
     - Cosine of cocoon opening angle
     - --
   * - ``log10_kappa``
     - Gray opacity (log10)
     - log10(cm^2/g)

Other Models
""""""""""""

**TDEAnalyticalModel**

Tidal disruption event with ``t^{-5/3}`` fallback accretion rate and
Arnett-style diffusion.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Units
   * - ``log10_l0``
     - Luminosity at 1 second (log10)
     - log10(erg/s)
   * - ``t_0_turn``
     - Turn-on time
     - days
   * - ``log10_mej``
     - Ejecta mass (log10)
     - log10(Msun)
   * - ``log10_vej``
     - Ejecta velocity (log10)
     - log10(km/s)
   * - ``log10_kappa``
     - Opacity (log10)
     - log10(cm^2/g)
   * - ``log10_kappa_gamma``
     - Gamma-ray opacity (log10)
     - log10(cm^2/g)

**SALT3Model**

Type Ia supernova spectral-template model using the SALT3 SED framework.
Requires the ``jax-supernovae`` package (``pip install jax-supernovae``).

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Parameter
     - Description
     - Units
   * - ``log10_x0``
     - SALT3 amplitude (log10)
     - --
   * - ``x1``
     - SALT3 stretch
     - --
   * - ``c``
     - SALT3 color
     - --
   * - ``t0``
     - Time of B-band maximum (observer-frame)
     - days

.. note::

   Unlike all other analytical models, ``SALT3Model`` expects times in
   the observer frame, not the source frame.


Writing a Custom Model
^^^^^^^^^^^^^^^^^^^^^^

To add a new analytical model, subclass ``AnalyticalModel`` and implement
``compute_log10_lbol_rphot``:

.. code:: python

    import jax.numpy as jnp
    from fiesta.inference.analytical_models import AnalyticalModel

    class MyModel(AnalyticalModel):
        parameter_names = ["log10_luminosity", "log10_radius"]

        def __init__(self, filters, times=None):
            super().__init__(filters=filters, times=times)

        def compute_log10_lbol_rphot(self, x, t_days):
            # Return (log10_Lbol [erg/s], log10_Rphot [cm]) at each t
            log10_L = jnp.full_like(t_days, x["log10_luminosity"])
            log10_R = jnp.full_like(t_days, x["log10_radius"])
            return log10_L, log10_R

The base class handles the blackbody SED, filter integration, redshift
correction, and distance modulus automatically.  The ``predict`` method
is JIT-compiled and differentiable out of the box.

For phenomenological models that parameterize magnitudes directly (rather
than bolometric luminosity), subclass ``PhenomenologicalModel`` instead
and implement ``compute_shape``.
