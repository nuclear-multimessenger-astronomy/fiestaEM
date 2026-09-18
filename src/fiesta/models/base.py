import abc

import jax
import jax.numpy as jnp
from jaxtyping import Array

from fiesta.logging import logger
from fiesta.filters import Filter

class FiestaModel(abc.ABC):
    name: str
    parameter_names: list[str]
    times: Array | None          # source-frame days
    filters: list[str]
    Filters: list[Filter]
    
    def __init__(self, name: str, filters, times=None):
        self.name = name
        self.filters, self.Filters = [], []
        self.times = jnp.asarray(times) if times is not None else None
        self.add_filters(filters)
    
    def add_filters(self, filters):
        if isinstance(filters, str) or isinstance(filters, Filter):
           filters = [filters]
        
        for filt in filters:
            if isinstance(filt, str):
                F = Filter(filt)
            elif isinstance(filt, Filter):
                F = filt
            else:
                raise TypeError(f"Filter needs to be a string or a Filter object.")

            if hasattr(self, "nus"):
                if F.nus[0]<self.nus[0] or F.nus[-1]>self.nus[-1]:
                    logger.warning(f"Filter {F.name} outside of frequency range of {self.name} surrogate. Not adding to the filter list.")
                    continue

            if F.name not in self.filters:
                self.filters.append(F.name)
                self.Filters.append(F)

        self._on_filters_changed()

    def _on_filters_changed(self) -> None:
        """Hook for subclasses that need to react to filter changes (e.g. rebuild a nu-grid)."""

    def add_name(self, x: Array):
        "Turns an unnamed array into a dictionary."
        return dict(zip(self.parameter_names, x))

    @abc.abstractmethod
    def predict(self, x: dict[str, Array]) -> tuple[Array, dict[str, Array]]:
        raise NotImplemented

    def predict_abs_mag(self, x: dict[str, Array]) -> tuple[Array, dict[str, Array]]:
        x = dict(x) # copy to avoid overwrite the caller's dictionary
        x["luminosity_distance"] = 1e-5
        x["redshift"] = 0.
        return self.predict(x)

    def vpredict(self, X: dict[str, Array]) -> tuple[Array, dict[str, Array]]:
        """
        Vectorized prediction function to calculate the apparent magnitudes for several inputs x at the same time.
        """
        return jax.vmap(self.predict)(X)

    def __repr__(self) -> str:
        return self.name