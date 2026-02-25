"""
Adapter classes that bridge fiesta's inference API with external samplers.
Currently provides wrappers for the jesterTOV sampler interface
"""

from fiesta.inference.prior_dict import ConstrainedPrior
from fiesta.inference.likelihood import EMLikelihood


class FiestaJesterPrior:
    """Wrap a fiesta ConstrainedPrior to satisfy the jesterTOV Prior interface.

    jesterTOV samplers look for ``parameter_names`` (not ``naming``).
    We expose it as a property and forward ``sample`` / ``log_prob``.
    """

    def __init__(self, fiesta_prior: ConstrainedPrior) -> None:
        self._prior = fiesta_prior
        # jesterTOV uses .parameter_names; fiesta uses .naming
        self.parameter_names = list(fiesta_prior.naming)

    @property
    def n_dim(self) -> int:
        return len(self.parameter_names)

    def sample(self, rng_key, n_samples: int) -> dict:
        return self._prior.sample(rng_key, n_samples)

    def log_prob(self, x: dict) -> float:
        return self._prior.log_prob(x)


class FiestaJesterLikelihood:
    """Wrap a fiesta EMLikelihood to satisfy the jesterTOV LikelihoodBase interface.

    jesterTOV calls ``likelihood.evaluate(params_dict)``.
    Fiesta's Fiesta class calls ``prior.transform(params)`` before the
    likelihood; we replicate that here so the adapter is fully general.
    """

    def __init__(
        self,
        fiesta_likelihood: EMLikelihood,
        fiesta_prior: ConstrainedPrior,
    ) -> None:
        self._likelihood = fiesta_likelihood
        self._prior = fiesta_prior

    def evaluate(self, params: dict) -> float:
        # apply the prior's transform (identity for KN, but kept for generality)
        transformed = self._prior.transform(params)
        return self._likelihood.evaluate(transformed)
