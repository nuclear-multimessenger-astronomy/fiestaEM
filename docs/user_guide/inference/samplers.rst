Samplers
--------

There are currently three different samplers that are directly integrated within fiesta's API.
Of course, the fiesta priors and likelihood functions could also be used with different samplers, but direct use through the ``Fiesta`` sampling class is only possible with either ``flowmc``, ``blackjax-smc``, or ``numpyro-svi``.
To decide which sampler to use, use the ``sampler`` argument when initializing the ``Fiesta`` class. By default, this will be ``flowmc``.
You can also pass additional sampling kwargs to the ``Fiesta`` class, if you do not wanna use their standard settings.

We briefly present the different samplers and their optional arguments here.


``flowmc`` (default)
^^^^^^^^^^^^^^^^^^^^

flowMC is an MCMC sampler that combines local Metropolis-adjusted Langevin algorithm (MALA) steps with a normalizing flow that is trained on the fly for efficient global proposals. 
This significantly improves sampling efficiency for complicated posterior distributions while still sampling the true posterior distribution.
Details can be found in the flowMC `paper <https://arxiv.org/abs/2211.06397>`_ and the `documentation <https://gw-jax-team.github.io/flowMC/stable/>`_.

The following keyword arguments can be passed to ``Fiesta(..., sampler="flowmc", **sampling_kwargs)``:

.. list-table::
   :header-rows: 1
   :widths: 25 60 15

   * - Argument
     - Description
     - Default
   * - ``n_chains`` 
     - Number of parallel MCMC chains. 
     - ``500``
   * - ``n_local_steps`` 
     - Number of local MALA steps per sampling loop. 
     - ``50``
   * - ``n_global_steps`` 
     - Number of global normalizing-flow proposals per sampling loop. 
     - ``200``
   * - ``n_training_loops``
     - Number of training sampling loops.
     - ``20``
   * - ``n_production_loops``
     - Number of production sampling loops after training.
     - ``15``
   * - ``n_epochs``
     - Number of epochs used to train the normalizing flow during each training loop.
     - ``100``
   * - ``rq_spline_n_layers``
     - Number of rational quadratic spline coupling layers.
     - ``4``
   * - ``rq_spline_hidden_units``
     - Hidden layer sizes of the neural networks used in the normalizing flow.
     - ``[64, 64]``
   * - ``rq_spline_n_bins``
     - Number of bins in each rational quadratic spline transform.
     - ``8``
   * - ``mala_step_size``
     - Step size used by the local MALA sampler.
     - ``2e-3``
   * - ``learning_rate``
     - Learning rate for normalizing flow training.
     - ``4e-4``
   * - ``n_max_examples``
     - Maximum number of training samples stored for training the flow.
     - ``10000``
   * - ``n_NFproposal_batch_size``
     - Batch size for generating normalizing-flow proposals.
     - ``10000``
   * - ``chain_batch_size``
     - Number of chains processed simultaneously.
     - ``100``
   * - ``batch_size``
     - Mini-batch size used during normalizing flow optimization.
     - ``10000``
   * - ``verbose``
     - Print progress information during sampling.
     - ``True``

For most applications, the default settings provide good performance. 
The parameters that are most commonly adjusted are ``n_chains``, ``n_training_loops``, ``n_production_loops``, and ``mala_step_size``.


.. code-block:: python

    sampler = Fiesta(
    ...,
    sampler="flowmc",
    n_chains=200,
    n_training_loops=30,
    n_production_loops=20,
    mala_step_size=1e-3,
    )

After sampling, fiesta returns the production samples as the posterior samples. 
If ``sampler_extra_output=True`` is passed to the ``Fiesta.save_results()`` method, additional diagnostic information from both the training and production phases (chains, log probabilities, acceptance rates, and training losses) are saved in two separate files ``results_training.npz`` and ``results_production.npz`` in the output directory.



``blackjax-smc``
^^^^^^^^^^^^^^^^

``blackjax-smc`` implements an adaptive Sequential Monte Carlo (SMC) sampler. 
Instead of directly sampling the posterior, the algorithm starts by drawing particles from the prior distribution and gradually transforms them into posterior samples through a sequence of intermediate distributions by increasing the inverse temperature on the likelihood function.
At each tempering stage, particles are rejuvenated using an MCMC kernel and resampled when necessary. 
An advantage of SMC methods is that they naturally provide an estimate of the Bayesian evidence.

The implementation in ``fiesta`` is based on the ``adaptive_tempered_smc`` algorithm from blackjax with automatic tuning of the tempering schedule. 
More details can be found in the ``blackjax`` `documentation <https://blackjax.readthedocs.io/>`_.

The following keyword arguments can be passed to ``Fiesta(..., sampler="blackjax-smc", **sampling_kwargs)``:

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Argument
     - Description
     - Default
   * - ``n_particles``
     - Number of particles used by the SMC sampler.
     - ``8000``
   * - ``target_ess``
     - Target relative effective sample size (ESS) used to adapt the tempering schedule. Larger values result in more tempering stages.
     - ``0.9``
   * - ``num_mcmc_steps``
     - Number of MCMC rejuvenation steps performed after each tempering stage.
     - ``10``
   * - ``kernel``
     - MCMC kernel used to rejuvenate particles. Currently only "random_walk" is implemented.
     - ``random_walk``
   * - ``random_walk_sigma``
     - Proposal scale of the random-walk kernel. Ignored for other kernels.
     - ``0.05``

For most applications, the default settings work well. The parameters that are most commonly adjusted are ``n_particles``, ``target_ess``, ``num_mcmc_steps``, and ``random_walk_sigma``.

.. code-block:: python

    sampler = Fiesta(
    ...,
    sampler="blackjax-smc",
    n_particles=10000,
    target_ess=0.95,
    num_mcmc_steps=20,
    random_walk_sigma=0.03,
    )

After sampling, ``fiesta`` returns the posterior samples. 
In addition, the estimated log-evidence is printed to the console. 
If ``sampler_extra_output=True`` is passed to the ``Fiesta.save_results()`` method, ``fiesta`` also saves a ``smc_metadata.json`` file containing diagnostic information.


``numpyro-svi``
^^^^^^^^^^^^^^^

``numpyro-svi`` performs Stochastic Variational Inference (SVI) using ``numpyro``. 
Instead of drawing exact samples from the posterior, SVI optimizes the parameters of a variational distribution to approximate the posterior by maximizing the Evidence Lower Bound (ELBO). 
Once optimization has converged, samples are drawn from the optimized variational distribution.
Here, we use a diagonal truncated normal distribution as variational distribution, meaning this method cannot represent correlations between different parameters. 

Summarized: This SVI method just serves as a cheap approximation method and should not be used to obtain accurate posterior samples.

The following keyword arguments can be passed to ``Fiesta(..., sampler="numpyro-svi", **sampling_kwargs)``:

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Argument
     - Description
     - Default
   * - ``num_iter``
     - Number of optimization iterations used to fit the variational posterior.
     - ``10000``
   * - ``step_size``
     - Learning rate of the Adam optimizer.
     - ``0.001``
   * - ``num_samples``
     - Number of posterior samples drawn from the optimized variational distribution.
     - ``1000``

For most applications, the default settings provide a good starting point. 
If the optimization has not converged, increasing ``num_iter`` is usually the first parameter to adjust. 
The ``step_size`` may also need tuning if the ELBO is unstable or converges slowly.

.. code-block:: python

    sampler = Fiesta(
    ...,
    sampler="numpyro-svi",
    num_iter=20000,
    step_size=5e-4,
    num_samples=5000,
    )

After optimization, ``fiesta`` draws ``num_samples`` samples from the learned variational distribution and returns them as posterior samples. 
The final ELBO is printed to the console. 
If ``sampler_extra_output=True`` is passed to the ``Fiesta.save_results()`` method, ``fiesta`` also saves a svi_metadata.json file containing diagnostic information.