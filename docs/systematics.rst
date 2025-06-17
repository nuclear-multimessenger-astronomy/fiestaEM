
Systematic error
----------------

Since neither the physical base model nor the surrogate reflects the physics of the transient perfectly, the likelihood function in fiesta uses a systematic uncertainty :math:`\sigma_{\rm sys}`. 

.. math::
    \mathcal{L}(\vec{\theta}|d) = - \frac{1}{2} \sum_{t_j} \biggl( \frac{(m(t_j) - m^{\star}(t_j, \vec{\!\theta}\,))^2}{\sigma(t_j)^2 + \sigma_{\text{sys}}^2} + \ln(2\pi (\sigma(t_j)^2 + \sigma_{\text{sys}}^2)) \biggr)

In fiesta, this is controlled through the ``EMLikelihood`` class. There are three ways :math:`\sigma_{\rm sys}` can be set up (see below). 
Which one of these modes is chosen depends on which ``_setup_sys_uncertainty_*`` method is called on the ``EMLikelihood`` instance. Upon initialization, the likelihood instance is in the "fixed" systematic uncertainty mode.
When doing inference, this is determined automatically depending on whether a systamics .yaml file is provided and whether the prior includes the ``sys_err`` variable.
The three modes are:

Fixed
^^^^^

In this case, :math:`\sigma_{\rm sys}` is just a constant determined through the ``error_budget`` keyword when initializing the likelihood function (defaults to 0.3 mag). 
In the inference this happens, when neither a systematics file is provided in the ``Fiesta`` sampler instance and the prior does not contain a ``sys_err`` parameter.

Free
^^^^

In this case, :math:`\sigma_{\rm sys}` is the same across all filters and times, but is sampled freely from a prior. This prior needs to be specified as ``sys_err`` in the prior.
When initializing a ``Fiesta`` sampler with such a prior, the likelihood function is updated accordingly, hence its value for ``error_budget`` will be overwritten.

Time-dependent
^^^^^^^^^^^^^^
In this case, the systematic uncertainty is sampled freely, is time-depdendent, and can also vary between filters.
For this to happen, the ``Fiesta`` sampler needs to be initialized with the argument ``systematics_file``, where ``systematics_file`` points to a .yaml file that will determine the specific setup.
This .yaml config should provide ``time_nodes``, i.e. the number of time nodes for which the systematic uncertainties are sampled. It should also specify a ``fiesta.inference.prior`` type and its parameters, except for the ``naming``.
For instance, to introduce four sampling parameters that represent the systematic uncertainty for all filters collectively, using four time nodes linearly spaced between 0.3 and 26 days, one could set up:

.. code:: yaml

    collective:
        time_nodes: 4
        time_range: linear 0.3 26
        prior: Uniform
        params: 
               xmin: 0.3 
               xmax: 1.0
    

Alternatively, if each filter should have their own systematic uncertainty parameters, the following works (here we omit the time_range argument, which causes the nodes to be spaced linearly between ``EMLikelihood.tmin`` and ``EMLikelihood.tmax``):

.. code:: yaml

    individual:
        time_nodes: 4
        prior: Uniform
        params: 
               xmin: 0.3 
               xmax: 1.0

We can also group several filters together:

.. code:: yaml

    group1:
        filters:
            - "X-ray-5keV"
            - "X-ray-1keV"
        time_nodes: 4
        time_range: linear 0.1 10
        prior: Uniform
        params:
               xmin: 0.3
               xmax: 2
    
    group2:
        filters:
            - "radio-3GHz"
            - "radio-6GHz"
        time_nodes: 2
        time_range: log 0.3 1000
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

This would mean the X-ray filters share their 4 systematic uncertainty parameters spaced linearly between 0.1 and 10 days, the radio filters are sampled with two separate systematic uncertainty filters spaced geometrically between 0.3 and 1000 days, and all remaining filters are sampled with a different set of four systematic uncertainty parameters that are linearly spaced between between ``EMLikelihood.tmin`` and ``EMLikelihood.tmax``.
If the data does not contain any remaining filters except ``X-ray-5keV, X-ray-1keV, radio-3GHz, radio-6GHz``, then ``remaining`` is ignored.