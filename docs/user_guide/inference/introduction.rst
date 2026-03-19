Basics
------

The inference technique implemented in ``fiesta`` is based on Bayesian statistics.
This means, given a series of magnitude measurements :math:`\{m(t_j)\}` for a transient, we want to know what the posterior distribution of the transient's parameters

.. math::
    p(\vec{\theta} | {m(t_j)}) = \frac{p(\{m(t_j)\}|\vec{\theta}) \pi(\vec{\theta})}{Z} = \frac{\mathcal{L}(\vec{\theta}|\{m(t_j)\}) \pi(\vec{\theta})}{Z}

Here, :math:`\pi(\vec{\theta})` is an agnostic prior function and :math:`\mathcal{L}(\vec{\theta} | {m(t_j)})` is the likelihood function that encodes which light curve :math:`m^{\star}(t_j, \vec{\!\theta}\,)` you would expected if the parameters :math:`\vec{\!\theta}\,)` you enter were the true ones.
The evidence :math:`Z` is the normalization factor for the posterior distribution.
We refer to :doc:`priors` and :doc:`likelihood` for a detailed explanation how these functions can be instantiated as objects in ``fiesta``.

The goal of the ``fiesta.inference`` API is to provide easy-to-use functions that implement the priors and likelihood. 
In the end, you can then combine them into the sampling API to numerically obtain the posterior

.. code:: Python

    from fiesta.inference import Fiesta
    
    fiesta = Fiesta(likelihood,
                    prior,
                    outdir)
    
    fiesta.sample(jax.random.key(42))
    fiesta.print_summary()
    fiesta.save_results()
    fiesta.plot_lightcurves()
    fiesta.plot_corner()

This will numerically sample the posterior distribution and save the results into outdir as an ``.npz`` file.
Additionally, it will save a ``.pkl`` file with the best fit parameters, produce a corner plot, and a plot of the data with the fitted light curves.
You can even get additional sampler output by setting ``fiesta.save(sampler_extra_output=True)`` to get some sampling metadata into ``outdir``.
If you run an injection, you can pass the true values to the ``plot_corner`` method as a dictionary.

Note that the ``Fiesta`` class also has a sampler keyword, which let's you choose between the different :doc:`samplers`.


