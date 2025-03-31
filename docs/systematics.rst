
Systematic error
----------------

Since neither the physical base model nor the surrogate reflects the physics of the transient perfectly, the likelihood function in fiesta uses a systematic uncertainty :math:`\sigma_{\rm sys}`. 

.. math::
    \mathcal{L}(\vec{\theta}|d) = - \frac{1}{2} \sum_{t_j} \biggl( \frac{(m(t_j) - m^{\star}(t_j, \vec{\!\theta}\,))^2}{\sigma(t_j)^2 + \sigma_{\text{sys}}^2} + \ln(2\pi (\sigma(t_j)^2 + \sigma_{\text{sys}}^2)) \biggr)

In fiesta, there are three ways :math:`\sigma_{\rm sys}` can be set up. For inference, these are determined on how the ``Fiesta`` sampling object is initialized.

Fixed
^^^^^

In this case, :math:`\sigma_{\rm sys}` is just a constant determined through the ``error_budget`` keyword when initializing the likelihood function (defaults to 0.3 mag). 
This is the default mode, when neither ``em_syserr`` is specified in the prior nor a systematic file is provided.

Free
^^^^

In this case, :math:`\sigma_{\rm sys}` is the same across all filters and times, but is sampled freely from a prior. This prior needs to be specified as ``em_syserr`` in the prior.
When initializing a ``Fiesta`` sampler with such a prior, the likelihood function is updated accordingly, hence its value for ``error_budget`` will be overwritten.

Time-dependent
^^^^^^^^^^^^^^

If the ``Fiesta`` sampler is initialized with the argument ``systematics_file``, then the systematic uncertainty is determined through the setup specified in the .yaml file ``systematics_file`` points to.
This config should provide ``time_nodes``, i.e. the number of time nodes for which the systematic uncertainties are sampled. It should also specify a ``fiesta.inference.prior`` type and its parameters, except for the ``naming``.
For instance, to introduce four sampling parameters that represent the systematic uncertainty at four time nodes (the systematic error for data points in between is determined through linear interpolation) for all filters collectively, one could set up:

.. code:: yaml

    collective:
        time_nodes: 4
        prior: Uniform
        params: 
               xmin: 0.3 
               xmax: 1.0
    

Alternatively, if each filter should have their own systematic uncertainty parameters, the following works:

.. code:: yaml

    individual:
        time_nodes: 4
        prior: Uniform
        params: 
               xmin: 0.3 
               xmax: 1.0

We can also group several filters together:

..code:: yaml

    group1:
        filters:
            - "X-ray-5keV"
            - "X-ray-1keV"
        time_nodes: 4
        prior: Uniform
        params:
               xmin: 0.3
               xmax: 2
    
    group2:
        filters:
            - "radio-3GHz"
            - "radio-6GHz"
        time_nodes: 2
        prior: LogUniform
        params:
               xmin: 0.3
               xmax: 1.0
    
    remaining:
        time_nodes: 4
        prior: Uniform
        params:
               xmin: 0.3
               xmax: 1.0

This would mean the X-ray filters share their 4 systematic uncertainty parameters, the radio filters are sampled with two separate systematic uncertainty filters, and all remaining filters are sampled with a different set of four systematic uncertainty parameters.
If the data does not contain any remaining filters except ``X-ray-5keV, X-ray-1keV, radio-3GHz, radio-6GHz``, then ``remaining`` is ignored.