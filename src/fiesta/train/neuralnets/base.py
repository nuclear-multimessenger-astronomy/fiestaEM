"""Base class for the neural network API"""
import pickle

from jaxtyping import Float, Array
from flax.training.train_state import TrainState

from .utils import serialize

class NN:
    """
    Abstract base class for the NN architecture wrappers of flax neural networks.
    """

    def train_loop(
        self,       
        train_X: Float[Array, "n_batch_train ndim_input"], 
        train_y: Float[Array, "n_batch_train ndim_output"],
        val_X: Float[Array, "n_batch_val ndim_output"] = None, 
        val_y: Float[Array, "n_batch_val ndim_output"] = None,
        verbose: bool = True
    ) -> tuple[TrainState, Array, Array]:
        raise NotImplementedError

    def save_model(self, outfile: str) -> None:
        """
        Serialize and save the model to a file.

        Raises:
            ValueError: If the provided file extension is not .pkl or .pickle.

        Args:
            outfile (str): The pickle file to which we save the serialized model.
        """

        if not outfile.endswith(".pkl") and not outfile.endswith(".pickle"):
            raise ValueError("For now, only .pkl or .pickle extensions are supported.")

        serialized_dict = serialize(self.trained_state, self.config)
        with open(outfile, 'wb') as handle:
            pickle.dump(serialized_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)