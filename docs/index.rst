fiesta documentation
====================

**F**\ast **I**\nference of **E**\lectromagnetic **S**\ignals and **T**\ransients with j\ **A**\x.

``fiesta`` is a JAX-based Python library for training machine-learning surrogates for
kilonova and gamma-ray burst afterglow models, and for performing fast Bayesian inference
on photometric lightcurve data from astronomical transients.

Recently, some analytical models have also been made availabe that can be used for light curve fitting.

.. note::

   **Documentation is work in progress!** Some sections may be incomplete or under active development. We appreciate your patience as we improve the documentation.
   Please contact us with any questions or issues you might encounter.


.. grid:: 2

   .. grid-item-card:: Installation
      :link: quickstart/installation
      :link-type: doc

      How to install the package.

   .. grid-item-card:: Surrogates
      :link: quickstart/surrogates
      :link-type: doc

      Learn about the two types of surrogate models (FluxModel and LightcurveModel)
      and how to load built-in and custom surrogates.

.. grid:: 2

   .. grid-item-card:: Training
      :link: user_guide/training/introduction
      :link-type: doc

      How to prepare training data and train your own surrogate models.

   .. grid-item-card:: Inference
      :link: user_guide/inference/introduction
      :link-type: doc

      How to perform Bayesian light curve fitting using the fiesta functionalities.

.. grid:: 2

   .. grid-item-card:: API Reference
      :link: api/fiesta
      :link-type: doc

      Full auto-generated API documentation for all modules.

   .. grid-item-card:: Citation
      :link: developer_guide/citing
      :link-type: doc

      If you use ``fiesta``, consider citing our paper 📝


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Quickstart

   quickstart/installation
   quickstart/surrogates
   quickstart/filters
   quickstart/analytical_models


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide
   
   user_guide/index
   user_guide/training/index
   user_guide/inference/index

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
   developer_guide/citing
