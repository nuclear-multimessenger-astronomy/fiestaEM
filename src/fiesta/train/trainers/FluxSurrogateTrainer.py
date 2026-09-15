"""API class to train machine-learning surrogates on """

import dill
import os
import pickle

import jax
from jaxtyping import Array, Float, Int
import numpy as np

import matplotlib.pyplot as plt

from fiesta.logging import logger
from fiesta.train import DataLoader
import fiesta.train.neuralnets as fiesta_nn


#####################
# FLUX TRAINING API #
#####################

class FluxSurrogateTrainer:
    """Training API class for training a surrogate model that predicts a spectral flux density array."""

    surrogte_name: str
    data: DataLoader
    outdir: str
    network: fiesta_nn.NN
    model_type: str

    def __init__(
            self,
            surrogate_name: str,
            data: DataLoader,
            outdir: str,
            network: fiesta_nn.NN,
            conversion: str = None,
            plots_dir: str = None,
            save_preprocessed_data: bool = False
        ) -> None:
        """
        Training API class for training a surrogate model that predicts a spectral flux density array.
        Supports different NN architectures through ``model_type``: ``"MLP"`` (trained on PCA coefficients
        of the training data) or ``"CVAE"`` (trained on a down-sampled flux image).
        Initializing will read the data with the DataLoader class; preprocessing happens once ``fit()`` is called.
        To write the surrogate model to file, the save() method is to be used, which will create two pickle files (one for the metadata, one for the neural network).

        Args:
            surrogate_name (str): Name of the model to be trained. Will be used when saving metadata and model to file.
            data (DataLoader): DataLoader class instance that will be used to read the data from the .h5 file in outdir and preprocess it.
            outdir (str): Directory where the NN and its metadata will be written to file.
            network (NN): Neural network to train.
            conversion (str): references how to convert the parameters for the training. Defaults to None, in which case it's the identity.
            plots_dir (str): Directory where the loss curves will be plotted. If ``None``, plots will be saved to ``outdir``. Defaults to None.
            save_preprocessed_data (bool): Whether the preprocessed training and validation data will be written to file. Defaults to ``False``.
        """

        self.surrogate_name = surrogate_name
        self.data = data

        self.conversion = conversion

        # Check if directories exists, otherwise, create:
        self.outdir = outdir
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.plots_dir = plots_dir
        if self.plots_dir is None:
            self.plots_dir = self.outdir
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

        self.save_preprocessed_data = save_preprocessed_data

        self.data.print_loaded_data_info()

        self.network = network
        self.model_type = type(network).__name__

    def __repr__(self) -> str:
        return f"FluxSurrogateTrainer({self.surrogate_name})"

    # -------
    # FITTING
    # -------

    def fit(
            self,
            verbose: bool = True
        ) -> None:
        """
        Method used to train the NN on the training data.

        Args:
            verbose (bool, optional): Whether the train and validation loss is printed to terminal in certain intervals. Defaults to True.
        """

        # Preprocess raw training data
        (   
            train_X, 
            train_y, 
            val_X, 
            val_y, 
            self.X_scaler, 
            self.y_scaler
                            ) = self.network.preprocess_data(self.data, self.conversion)

        # If desired, save preprocessed data
        if self.save_preprocessed_data:
            self._save_preprocessed_data(train_X, train_y, val_X, val_y)

        # Perform training loop
        state, train_losses, val_losses = self.network.train_loop(train_X, train_y, val_X, val_y, verbose=verbose)

        # Plot losses
        self.plot_learning_curve(train_losses, val_losses)

    def save(self) -> None:
        """
        Save the trained model and all the metadata to the outdir.
        The meta data is saved as a pickled dict to be read by ``fiesta.models.surrogate_models.Surrogate``.
        The NN is saved as a pickled serialized dict using the ``NN.save_model`` method.
        """
        # Save the metadata
        meta_filename = os.path.join(self.outdir, f"{self.surrogate_name}_metadata.pkl")
        
        save = {}
        save["times"] = self.data.times
        save["nus"] = self.data.nus
        save["parameter_names"] = self.data.parameter_names
        save["parameter_distributions"] = self.data.parameter_distributions
        save["X_scaler"] = self.X_scaler
        save["y_scaler"] = self.y_scaler
        save["model_type"] = self.model_type

        with open(meta_filename, "wb") as meta_file:
            dill.dump(save, meta_file)
        
        # Save the NN
        self.network.save_model(outfile=os.path.join(self.outdir, f"{self.surrogate_name}.pkl"))
    
    def _save_preprocessed_data(self, train_X, train_y, val_X, val_y) -> None:
        logger.info("Saving preprocessed data . . .")
        np.savez(
            os.path.join(self.outdir, f"{self.surrogate_name}_preprocessed_data.npz"), 
            train_X=train_X, 
            train_y=train_y, 
            val_X=val_X, 
            val_y=val_y
        )
        logger.info("Saving preprocessed data . . . done")

    # --------
    # PLOTTING
    # --------
    
    def plot_learning_curve(self, train_losses, val_losses):
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = np.arange(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, "-", lw=1.0, label="Train", color="red")
        ax.plot(epochs, val_losses, "-", lw=1.0, label="Validation", color="blue")

        # Mark best validation epoch
        best_idx = np.argmin(val_losses)
        ax.axvline(best_idx + 1, color="blue", ls="--", alpha=0.4, lw=0.8)
        ax.annotate(f"Best val @ {best_idx + 1}", 
                    xy=(0.6, 0.8), xycoords="figure fraction",
                    fontsize=11, color="blue", alpha=0.7,
                    xytext=(10, 10), textcoords="offset points")

        ax.legend(fontsize=11, fancybox=False, framealpha=1)
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.set_yscale("log")
        ax.set_title("Learning curves", fontsize=16)
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(self.plots_dir, f"learning_curves_{self.surrogate_name}.png"),
                    bbox_inches="tight", dpi=150)
        plt.close(fig)
    
    def plot_example_lc(self, filters: list[str]):
        from fiesta.models import FluxSurrogate
        lc_model = FluxSurrogate(self.surrogate_name, filters, self.outdir)

        # load last sample from validation data
        X, y = self.data.load_from_file("val", index=-1)

        y = y.reshape(len(self.data.nus), len(self.data.times))
        mJys_val = np.power(10, y)
        params = dict(zip(self.data.parameter_names, X.flatten() ))
        # compare at redshift 0 / the flux training reference distance (10 pc)
        _, mag_predict = lc_model.predict_abs_mag(params)
        mag_val = {Filt.name: Filt.get_mag(mJys_val, self.data.nus) 
                        for Filt in lc_model.Filters}

        for filt in lc_model.Filters:
    
            plt.plot(lc_model.times, mag_val[filt.name], color = "red", label="Base model")
            plt.plot(lc_model.times, mag_predict[filt.name], color = "blue", label="Surrogate prediction")
            upper_bound = mag_predict[filt.name] + 1
            lower_bound = mag_predict[filt.name] - 1
            plt.fill_between(lc_model.times, lower_bound, upper_bound, color='blue', alpha=0.2)
        
            plt.ylabel(f"mag for {filt.name}")
            plt.xlabel("$t$ in days")
            plt.legend()
            plt.gca().invert_yaxis()
            plt.xscale('log')
            plt.xlim(lc_model.times[0], lc_model.times[-1])

            plt.savefig(
                os.path.join(self.plots_dir, f"{self.surrogate_name}_{filt.name}_example.png"), 
                bbox_inches="tight"
            )
            plt.close()