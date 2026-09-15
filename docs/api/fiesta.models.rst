fiesta.models
=============

Model classes implemented in fiesta: surrogate (neural-network) models,
analytical (physics-based) models, and combinations thereof. All of them
share the common ``FiestaModel`` interface (``name``, ``parameter_names``,
``times``, ``filters``, ``predict()``).

Base Interface
--------------

.. automodule:: fiesta.models.base
   :members:
   :undoc-members:
   :show-inheritance:

Surrogate Models
----------------

.. automodule:: fiesta.models.surrogate_models
   :members:
   :undoc-members:
   :show-inheritance:

Combined Models
----------------

.. automodule:: fiesta.models.combined_model
   :members:
   :undoc-members:
   :show-inheritance:

Analytical Models
------------------

.. automodule:: fiesta.models.analytical_models.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fiesta.models.analytical_models.phenomenological_models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fiesta.models.analytical_models.supernova_models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fiesta.models.analytical_models.kilonova_models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fiesta.models.analytical_models.shock_powered_models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fiesta.models.analytical_models.tde_models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fiesta.models.analytical_models.salt3_models
   :members:
   :undoc-members:
   :show-inheritance:
