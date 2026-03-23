Installation
------------

``fiesta`` can be installed from PyPI:

.. code-block:: bash

   pip install fiestaEM

Or directly from source for the latest development version:

.. code-block:: bash

   git clone https://github.com/nuclear-multimessenger-astronomy/fiestaEM.git
   pip install -e .

Note, that by default only the cpu version of `jax` is installed. If you want to use GPU acceleration, run:

.. code-block:: bash

   pip install fiestaEM[gpu]

This requires CUDA13, but is important if you want to train surrogates or use GPU-accelerated inference.

When you want to create your own GRB afterglow training data set install

.. code-block:: bash

   pip install fiestaEM[grb]


To obtain a set of recommended built-in surrogates (see below), additionally run

.. code:: bash
    
    python -c "from fiesta.surrogates import download_recommended_surrogates; download_recommended_surrogates()"
