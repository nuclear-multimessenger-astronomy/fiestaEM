fiesta.train
============

Components for training surrogate models.

Trainers
--------

``FluxSurrogateTrainer`` is the actively maintained training path: it trains a single
spectral-flux surrogate covering all filters at once. ``LightcurveSurrogateTrainer``
(and its ``SVDTrainer`` subclass) predates it and is kept only so that already-trained
``fiesta.models.surrogate_models.LightcurveSurrogate`` models can still be reproduced
or retrained; it is deprecated in favor of ``FluxSurrogateTrainer``.

.. automodule:: fiesta.train.trainers.FluxSurrogateTrainer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fiesta.train.trainers.LightcurveSurrogateTrainer
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: train_X, train_y, val_X, val_y

Data
----

.. automodule:: fiesta.train.DataLoader
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fiesta.train.AfterglowData
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fiesta.train.utils
   :members:
   :undoc-members:
   :show-inheritance:

Neural Networks
---------------

``fiesta.train.neuralnets`` wraps the raw Flax network definitions in
``fiesta.train.nn_architectures`` with a common training-loop interface (the ``NN``
base class) and a shared configuration object (``NeuralnetConfig``).

.. automodule:: fiesta.train.neuralnets.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fiesta.train.neuralnets.mlp
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fiesta.train.neuralnets.cvae
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fiesta.train.neuralnets.utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fiesta.train.nn_architectures
   :members:
   :undoc-members:
   :show-inheritance:

Benchmarking
------------

.. automodule:: fiesta.train.Benchmarker
   :members:
   :undoc-members:
   :show-inheritance:
