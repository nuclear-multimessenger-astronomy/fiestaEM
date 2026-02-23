fiesta
======

**F**\ast **I**\nference of **E**\lectromagnetic **S**\ignals and **T**\ransients with j\ **A**\x.

``fiesta`` is a JAX-based Python library for training machine-learning surrogates for
kilonova and gamma-ray burst afterglow models, and for performing fast Bayesian inference
on photometric lightcurve data from such events.

.. grid:: 2

   .. grid-item-card:: Getting Started
      :link: quickstart
      :link-type: doc

      New to ``fiesta``? Start here for a brief introduction and first steps.

   .. grid-item-card:: Surrogates
      :link: surrogates
      :link-type: doc

      Learn about the two types of surrogate models (FluxModel and LightcurveModel)
      and how to load built-in and custom surrogates.

.. grid:: 2

   .. grid-item-card:: Training
      :link: overview/training
      :link-type: doc

      How to prepare training data and train your own surrogate models.

   .. grid-item-card:: Inference
      :link: overview/inference
      :link-type: doc

      How to set up and run Bayesian parameter estimation with ``fiesta``.

.. grid:: 2

   .. grid-item-card:: Systematic Errors
      :link: systematics
      :link-type: doc

      Configure fixed, free, or time-dependent systematic uncertainties in the
      likelihood.

   .. grid-item-card:: API Reference
      :link: api/fiesta
      :link-type: doc

      Full auto-generated API documentation for all modules.

Installation
------------

``fiesta`` can be installed from PyPI:

.. code-block:: bash

   pip install fiestaEM

Or directly from source for the latest development version:

.. code-block:: bash

   git clone https://github.com/nuclear-multimessenger-astronomy/fiestaEM.git
   pip install -e .

For GPU acceleration (requires CUDA 12):

.. code-block:: bash

   pip install fiestaEM[gpu]

For GRB afterglow support:

.. code-block:: bash

   pip install fiestaEM[grb]

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   quickstart
   surrogates
   systematics
   filters
   training_data

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Overview

   overview/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   api/fiesta

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Developer Guide

   developer_guide/index
   citing
