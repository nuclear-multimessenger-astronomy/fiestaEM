"""Method to train the surrogate models"""

import os
import numpy as np

import jax
import jax.numpy as jnp 
from jaxtyping import Array, Float, Int

from sklearn.decomposition import PCA
from fiesta.utils import MinMaxScalerJax
from fiesta import utils
from fiesta import conversions
from fiesta.constants import days_to_seconds, c
from fiesta import models_utilities
import fiesta.train.neuralnets as fiesta_nn

import matplotlib.pyplot as plt
import pickle
from typing import Callable
import tqdm
from multiprocessing import Pool


import afterglowpy as grb
from PyBlastAfterglowMag.wrappers import run_grb

class FluxTrainer:
    """Abstract class for training a collection of surrogate"""
    
    name: str
    outdir: str
    parameter_names: list[str]
    
    preprocessing_metadata: dict[str, dict[str, float]]
    
    X_raw: Float[Array, "n_batch n_params"]
    y_raw: dict[str, Float[Array, "n_batch n_times"]]
    
    X: Float[Array, "n_batch n_input_surrogate"]
    y: dict[str, Float[Array, "n_batch n_output_surrogate"]]
    
    trained_states: dict[str, fiesta_nn.TrainState]
    
    def __init__(self, 
                 name: str,
                 outdir: str,
                 plots_dir: str = None, 
                 ) -> None:
        
        self.name = name
        self.outdir = outdir
        # Check if directories exists, otherwise, create:
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        
        self.plots_dir = plots_dir
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
        
       
        # To be loaded by child classes
        self.parameter_names = None
                
        self.preprocessing_metadata = {}
        
        self.train_X_raw = None
        self.train_y_raw = None

        self.val_X_raw = None
        self.val_y_raw = None

        self.weights = None

    def __repr__(self) -> str:
        return f"FluxTrainer(name={self.name})"
    
    def preprocess(self):
        
        print("Preprocessing data by minmax scaling . . .")
        self.X_scaler = MinMaxScalerJax()
        self.X = self.X_scaler.fit_transform(self.train_X_raw)
        
        self.y_scaler = MinMaxScalerJax()
        self.y = self.y_scaler.fit_transform(self.train_y_raw)
            
        # Save the metadata
        self.preprocessing_metadata["X_scaler_min"] = self.X_scaler.min_val 
        self.preprocessing_metadata["X_scaler_max"] = self.X_scaler.max_val
        print("Preprocessing data . . . done")
    
    def fit(self,
            config: fiesta_nn.NeuralnetConfig = None,
            key: jax.random.PRNGKey = jax.random.PRNGKey(0),
            verbose: bool = True):
        """
        The config controls which architecture is built and therefore should not be specified here.
        
        Args:
            config (nn.NeuralnetConfig, optional): _description_. Defaults to None.
        """
        
        # Get default choices if no config is given
        if config is None:
            config = fiesta_nn.NeuralnetConfig()
        self.config = config
            
        input_ndim = len(self.parameter_names)

           
        # Create neural network and initialize the state
        net = fiesta_nn.MLP(layer_sizes=config.layer_sizes)
        key, subkey = jax.random.split(key)
        state = fiesta_nn.create_train_state(net, jnp.ones(input_ndim), subkey, config)
        
        # Perform training loop
        state, train_losses, val_losses = fiesta_nn.train_loop(state, config, self.train_X, self.train_y, self.val_X, self.val_y, verbose=verbose)
        # Plot and save the plot if so desired
        if self.plots_dir is not None:
            plt.figure(figsize=(10, 5))
            ls = "-o"
            ms = 3
            plt.plot([i+1 for i in range(len(train_losses))], train_losses, ls, markersize=ms, label="Train", color="red")
            plt.plot([i+1 for i in range(len(val_losses))], val_losses, ls, markersize=ms, label="Validation", color="blue")
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("MSE loss")
            plt.yscale('log')
            plt.title("Learning curves")
            plt.savefig(os.path.join(self.plots_dir, f"learning_curves_{self.name}.png"))
            plt.close()     
       
        self.trained_state = state
        
    def save(self):
        """
        Save the trained model and all the used metadata to the outdir.
        """
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        meta_filename = os.path.join(self.outdir, f"{self.name}_metadata.pkl")
        
        """
        # Save the meta data
        meta_filename = os.path.join(self.outdir, f"{self.name}_metadata.pkl")
        if os.path.exists(meta_filename):
            with open(meta_filename, "rb") as meta_file:
                save = pickle.load(meta_file)
            if not np.array_equal(save["times"], self.times): # check whether the metadata from previously trained filters agrees
                raise Exception(f"The time array needs to coincide with the time array for previous filters: {save['times']}")
            if not np.array_equal(save["parameter_names"], self.parameter_names):
                 raise Exception(f"The parameters need to coincide with the parameters for previous filters: {save['parameter_names']}")
        else:
            save = {}
        
        """
        save = {}

        save["times"] = self.times
        save["nus"] = self.nus
        save["parameter_names"] = self.parameter_names # TODO: see if we can save the jet_type here somewhat more self-consistently
        save["meta_data"] = self.preprocessing_metadata # TODO: maybe split the preprocessing_metadata thing here up into its keys

        with open(meta_filename, "wb") as meta_file:
            pickle.dump(save, meta_file)
        
        # Save the NN
        model = self.trained_state
        fiesta_nn.save_model(model, self.config, out_name=self.outdir + f"{self.name}.pkl")
    
    def _save_preprocessed_data(self):
        print("Saving preprocessed data . . .")
        np.savez(os.path.join(self.outdir, "afterglowpy_preprocessed_data.npz"), train_X=self.train_X, train_y= self.train_y, val_X = self.val_X, val_y = self.val_y)
        print("Saving preprocessed data . . . done")

class PCATrainer(FluxTrainer):

    def __init__(self,
                 name: str,
                 outdir: str,
                 data_manager,
                 n_pca: Int = 100,
                 plots_dir: str = None,
                 save_preprocessed_data: bool = False):

        super().__init__(name = name,
                       outdir = outdir,
                       plots_dir = plots_dir)

        self.n_pca = n_pca
        self.save_preprocessed_data = save_preprocessed_data
        self.data_manager = data_manager
        
        self.plots_dir = plots_dir
        if self.plots_dir is not None and not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
        

        self.data_manager.pass_data(self)
        self.preprocess()
    
        if save_preprocessed_data:
            self._save_preprocessed_data()
        
    def preprocess(self):

        # rescale the parameters to lie between 0 and 1
        X_scaler = MinMaxScalerJax()
        self.train_X = X_scaler.fit_transform(self.train_X_raw) # fit the scaler to the training data
        self.val_X = X_scaler.transform(self.val_X_raw) # transform the val data
        self.X_scaler = X_scaler


        #rescale data to lie between 0 and 1
        y_scaler = MinMaxScalerJax()
        self.train_y = y_scaler.fit_transform(self.train_y_raw)
        self.val_y = y_scaler.transform(self.val_y_raw)
        self.y_scaler = y_scaler

        self.preprocessing_metadata["X_scaler"] = self.X_scaler 
        self.preprocessing_metadata["y_scaler"] = self.y_scaler
            
        pca = PCA(n_components=self.n_pca)
        print(f"Fitting PCA model with {self.n_pca} components to the provided data.")
        self.train_y = pca.fit_transform(self.train_y)
        print(f"PCA model accounts for a share {np.sum(pca.explained_variance_ratio_)} of the total variance in the training data. This value is hopefully close to 1.")
        self.val_y = pca.transform(self.val_y)

        self.pca = pca
        self.preprocessing_metadata["pca"] = pca
        print("Preprocessing data . . . done")

        
    def load_parameter_names(self):
        raise NotImplementedError
        
    def load_times(self):
        raise NotImplementedError
       
    def load_raw_data(self):
        raise NotImplementedError



class DataManager:
    
    def __init__(self,
                 outdir: str,
                 n_training: Int, 
                 n_val: Int, 
                 tmin: Float,
                 tmax: Float,
                 numin: Float = 1e9,
                 numax: Float = 2.5e18, 
                 retrain_weights: dict = None):
        
        self.outdir = outdir
        self.n_training = n_training
        self.n_val = n_val

        self.tmin = tmin
        self.tmax = tmax
        self.numin = numin
        self.numax = numax

        self.retrain_weights = retrain_weights
        
        self.get_data_from_file()
        self.set_up_raw_data()
        self.select_raw_data()

    def get_data_from_file(self,)->None:
        with open(os.path.join(self.outdir, "afterglowpy_raw_data.pkl"), "rb") as file:
            data = pickle.load(file)

        self.parameter_distributions = data["parameter_distributions"]
        self.parameter_names = data["parameter_names"]

        self.times_data = data["times"]        
        self.nus_data = data["nus"]

        self.training_X_data = data["train_X_raw"]
        self.training_y_data = data["train_y_raw"]
        self.val_X_data = data["val_X_raw"]
        self.val_y_data = data["val_y_raw"]

    def set_up_raw_data(self,)->None:
        """Trims the stored data down to the time and frequency range desired for training."""
        
        if self.tmin<self.times_data.min() or self.tmax>self.times_data.max():
            print(f"\nWarning: provided time range {self.tmin, self.tmax} is too wide for the data stored in file. Using range {max(self.times_data.min(), self.tmin), min(self.times_data.max(), self.tmax)} instead.\n")
        time_mask = np.logical_and(self.times_data>=self.tmin, self.times_data<=self.tmax)
        self.times = self.times_data[time_mask]
        self.n_times = len(self.times)

        if self.numin<self.nus_data.min() or self.numax>self.nus_data.max():
            print(f"\nWarning: provided frequency range {self.numin, self.numax} is too wide for the data stored in file. Using range {max(self.nus_data.min(), self.numin), min(self.nus_data.max(), self.numax)} instead.\n")
        nu_mask = np.logical_and(self.nus_data>=self.numin, self.nus_data<=self.numax)
        self.nus = self.nus_data[nu_mask]
        self.n_nus = len(self.nus)
        
        mask = nu_mask[:, None] & time_mask
        mask = mask.flatten()
        self.train_X_raw = self.training_X_data
        self.train_y_raw = self.training_y_data[:,mask]
        self.val_X_raw = self.val_X_data
        self.val_y_raw = self.val_y_data[:,mask]
    
    def select_raw_data(self,)->None:
        if len(self.train_X_raw)<self.n_training:
            raise ValueError(f"Only {len(self.train_X_raw)} entries in file, not enough to train with {self.n_training} data points.")
        if len(self.val_X_raw)<self.n_val:
            raise ValueError(f"Only {len(self.val_X_raw)} entries in file, not enough to validate with {self.n_val} data points.")

        select = np.random.choice(range(len(self.training_X_data)), size = self.n_training, replace = False)
        self.train_X_raw, self.train_y_raw = self.train_X_raw[select], self.train_y_raw[select]
    
        select = np.random.choice(range(len(self.val_X_data)), size = self.n_val, replace = False)
        self.val_X_raw, self.val_y_raw = self.val_X_raw[select], self.val_y_raw[select]


    def pass_data(self, object):
        object.parameter_names = self.parameter_names
        object.train_X_raw = self.train_X_raw
        object.train_y_raw = self.train_y_raw
        object.val_X_raw = self.val_X_raw
        object.val_y_raw = self.val_y_raw
        object.times = self.times
        object.nus = self.nus


class AfterglowData:
    def __init__(self,
                 outdir: str,
                 jet_type: Int,
                 n_training: Int, 
                 n_val: Int,
                 n_test: Int,
                 parameter_distributions: dict,
                 n_pool: Int,
                 tmin: Int = 1,
                 tmax: Int = 1000,
                 n_times: Int = 100,
                 use_log_spacing: bool = True,
                 numin: Float = 1e9,
                 numax: Float = 2.5e18,
                 n_nu: Int = 256,
                 fixed_parameters: dict = {},
                 remake_data: bool = False) -> None:
        
        self.outdir = outdir
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.n_training = n_training
        self.n_val = n_val
        self.n_test = n_test

        if os.path.exists(self.outdir+"/afterglowpy_raw_data.pkl"):
            self._read_files()

        else:
            self.jet_type = jet_type
            if self.jet_type not in [-1,0,4]:
                raise ValueError(f"Jet type {jet_type} is not supported. Supported jet types are: [-1, 0, 4]")
    
            self.load_times(tmin, tmax, n_times, use_log_spacing)
            self.load_nus(numin, numax, n_nu)
            self.parameter_distributions = parameter_distributions
            self.parameter_names = list(parameter_distributions.keys())
        
        self.fixed_parameters = fixed_parameters
        self.n_pool = n_pool

        if remake_data:
            os.remove(self.outdir+"/afterglowpy_raw_data.pkl")

        self.get_raw_data()
        self.check_nans()
        if self.save_raw_data:
            self._save_to_file()


    def load_times(self, tmin, tmax, n_times, use_log_spacing):
        if use_log_spacing:
            times = np.logspace(np.log10(tmin), np.log10(tmax), num=n_times)
        else:
            times = np.linspace(tmin, tmax, num=n_times)
        self.times = times
    
    def load_nus(self, numin, numax, n_nu):
        self.nus = np.logspace(np.log10(numin), np.log10(numax), n_nu)
    
    def get_raw_data(self):
        self.save_raw_data = False
        if os.path.exists(self.outdir+"/afterglowpy_raw_data.pkl"):
            if self.n_training> len(self.train_X_raw):
                n = self.n_training - len(self.train_X_raw)
                print(f"Supplementing the afterglowpy training dataset on grid with {n} points.")
                new_training_X_raw, new_training_y_raw = self.create_raw_data(n)
                self.train_X_raw = np.concatenate((self.train_X_raw, new_training_X_raw))
                self.train_y_raw = np.concatenate((self.train_y_raw, new_training_y_raw))
                self.save_raw_data = True
            
            if self.n_val> len(self.val_X_raw):
                n = self.n_val - len(self.val_X_raw)
                print(f"Supplementing the afterglowpy validation dataset on {n} points within grid.")
                new_val_X_raw, new_val_y_raw = self.create_raw_data(n, training = False)
                self.val_X_raw = np.concatenate((self.val_X_raw, new_val_X_raw))
                self.val_y_raw = np.concatenate((self.val_y_raw, new_val_y_raw))
                self.save_raw_data = True
            
            if self.n_test> len(self.test_X_raw):
                n = self.n_test - len(self.test_X_raw)
                print(f"Supplementing the afterglowpy test dataset on grid with {n} points within grid.")
                new_test_X_raw, new_test_y_raw = self.create_raw_data(n, training = False)
                self.test_X_raw = np.concatenate((self.test_X_raw, new_test_X_raw))
                self.test_y_raw = np.concatenate((self.test_y_raw, new_test_y_raw))
                self.save_raw_data = True

        else:
            print("No data file found. Creating data from scratch.")
            self.train_X_raw, self.train_y_raw = self.create_raw_data(self.n_training)
            self.val_X_raw, self.val_y_raw = self.create_raw_data(self.n_val, training = False)
            self.test_X_raw, self.test_y_raw = self.create_raw_data(self.n_test, training = False)
            self.save_raw_data = True
    
    def check_nans(self,):
        # fixes any nans that remain from create_raw_data
        for xname, yname in zip(["train_X_raw", "val_X_raw", "test_X_raw"], ["train_y_raw", "val_y_raw", "test_y_raw"]):
            x, y = getattr(self, xname), getattr(self, yname)
            problematic = np.unique(np.where(np.isnan(y))[0])
            n = len(problematic)
            while n>0:
                self.save_raw_data = True
                x_replacement, y_replacement = self.create_raw_data(n)
                x[problematic] = x_replacement
                y[problematic] = y_replacement
                problematic = np.unique(np.where(np.isnan(y))[0])
                n = len(problematic)
            setattr(self, xname, x)
            setattr(self, yname, y)
    
    def _read_files(self,):

        with open(os.path.join(self.outdir, "afterglowpy_raw_data.pkl"), "rb") as file:
            data = pickle.load(file)
        self.jet_type = data["jet_type"]
        self.times = data["times"]
        self.nus = data["nus"]
        self.parameter_distributions = data["parameter_distributions"]
        self.parameter_names = data["parameter_names"]
        self.train_X_raw, self.train_y_raw = data["train_X_raw"], data["train_y_raw"]
        self.val_X_raw, self.val_y_raw = data["val_X_raw"], data["val_y_raw"]
        self.test_X_raw, self.test_y_raw = data["test_X_raw"], data["test_y_raw"]
        print("Loaded data from file, ignoring input arguments for jet type, times, and frequencies.")

    def create_raw_data(self, n, training = True):
        raise NotImplementedError

    def _save_to_file(self):

        save = {}
        save["times"] = self.times
        save["nus"] = self.nus
        save["jet_type"] = self.jet_type
        save["parameter_distributions"] = self.parameter_distributions
        save["parameter_names"]  = self.parameter_names

        save["train_X_raw"] = self.train_X_raw
        save["train_y_raw"] = self.train_y_raw
        save["val_X_raw"] = self.val_X_raw
        save["val_y_raw"] = self.val_y_raw
        save["test_X_raw"] = self.test_X_raw
        save["test_y_raw"] = self.test_y_raw

        print("Saving raw data . . .")
        with open(os.path.join(self.outdir, "afterglowpy_raw_data.pkl"), "wb") as outfile:
            pickle.dump(save, outfile)
        print("Saving raw data . . . done")

class AfterglowpyData(AfterglowData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def create_raw_data(self, n, training = True):
        """
        Create a grid of training data with specified settings.
        """
        # Create training data
        X_raw = np.empty((n, len(self.parameter_names)))
        y_raw = np.empty((n, len(self.times)*len(self.nus)))
        
        if training:
            for j, key in enumerate(self.parameter_distributions.keys()):
                a, b, distribution = self.parameter_distributions[key]
                if distribution == "uniform":
                    X_raw[:,j] = np.random.uniform(a, b, size = n)
                elif distribution == "loguniform":
                    X_raw[:,j] = np.exp(np.random.uniform(np.log(a), np.log(b), size = n))
        else:
            for j, key in enumerate(self.parameter_distributions.keys()):
                a, b, _ = self.parameter_distributions[key]
                X_raw[:, j] = np.random.uniform(a, b, size = n)
        
        if self.jet_type ==0:
            alphac_ind = self.parameter_names.index("alphaCore")
            thetaw_ind = self.parameter_names.index("thetaWing")
            mask = X_raw[:, alphac_ind]*X_raw[:, thetaw_ind] < 0.01
            X_raw[mask, alphac_ind] = 0.01/X_raw[mask, thetaw_ind]

        afgpy = RunAfterglowpy(self.jet_type, self.times, self.nus, X_raw, self.parameter_names, self.fixed_parameters)
        pool = Pool(processes=self.n_pool)
        jobs = [pool.apply_async(func=afgpy, args=(argument,)) for argument in range(len(X_raw))]
        pool.close()
        for Idx, job in enumerate(tqdm.tqdm(jobs)):
            try:
                idx, out = job.get()
                y_raw[idx] = out
            except:
                y_raw[Idx] = np.full(len(self.times)*len(self.nus), np.nan)
        return X_raw, y_raw


class PyblastafterglowData(AfterglowData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def create_raw_data(self, n, training = True):
        """
        Create a grid of training data with specified settings.
        """
        # Create training data
        X_raw = np.empty((n, len(self.parameter_names)))
        y_raw = np.empty((n, len(self.times)*len(self.nus)))
        np.random.seed(32)
        if training:
            for j, key in enumerate(self.parameter_distributions.keys()):
                a, b, distribution = self.parameter_distributions[key]
                if distribution == "uniform":
                    X_raw[:,j] = np.random.uniform(a, b, size = n)
                elif distribution == "loguniform":
                    X_raw[:,j] = np.exp(np.random.uniform(np.log(a), np.log(b), size = n))
        else:
            for j, key in enumerate(self.parameter_distributions.keys()):
                a, b, _ = self.parameter_distributions[key]
                X_raw[:, j] = np.random.uniform(a, b, size = n)
        pbag = RunPyblastafterglow(self.jet_type, self.times, self.nus, X_raw, self.parameter_names, self.fixed_parameters)
        breakpoint()
        #pool = Pool(processes=self.n_pool)
        #jobs = [pool.apply_async(func=pbag, args=(argument,)) for argument in range(len(X_raw))]
        #pool.close()
        #for job in tqdm.tqdm(jobs):
        #    idx, out = job.get()
        #    y_raw[idx] = out
            
        #return X_raw, y_raw


class RunAfterglowpy:
    def __init__(self, jet_type, times, nus, X, parameter_names, fixed_parameters = {}):
        self.jet_type = jet_type
        self.times = times
        self._times_afterglowpy = self.times * days_to_seconds # afterglowpy takes seconds as input
        self.nus = nus
        self.X = X
        self.parameter_names = parameter_names
        self.fixed_parameters = fixed_parameters


    def _call_afterglowpy(self,
                         params_dict: dict[str, Float]) -> Float[Array, "n_times"]:
        """
        Call afterglowpy to generate a single flux density output, for a given set of parameters. Note that the parameters_dict should contain all the parameters that the model requires, as well as the nu value.
        The output will be a set of mJys.

        Args:
            Float[Array, "n_times"]: The flux density in mJys at the given times.
        """
        
        # Preprocess the params_dict into the format that afterglowpy expects, which is usually called Z
        Z = {}
        
        Z["jetType"]  = params_dict.get("jetType", self.jet_type)
        Z["specType"] = params_dict.get("specType", 0)
        Z["z"] = params_dict.get("z", 0.0)
        Z["xi_N"] = params_dict.get("xi_N", 1.0)
            
        Z["E0"]        = 10 ** params_dict["log10_E0"]
        Z["n0"]        = 10 ** params_dict["log10_n0"]
        Z["p"]         = params_dict["p"]
        Z["epsilon_e"] = 10 ** params_dict["log10_epsilon_e"]
        Z["epsilon_B"] = 10 ** params_dict["log10_epsilon_B"]
        Z["d_L"]       = 3.086e19 # fix at 10 pc, so that AB magnitude equals absolute magnitude

        if "inclination_EM" in list(params_dict.keys()):
            Z["thetaObs"]  = params_dict["inclination_EM"]
        else:
            Z["thetaObs"]  = params_dict["thetaObs"]

        if self.jet_type == -1:
             Z["thetaCore"] = params_dict["thetaCore"]
        
        if self.jet_type == 0:
             Z["thetaWing"] = params_dict["thetaWing"]
             Z["thetaCore"] = params_dict["alphaCore"]*params_dict["thetaWing"]

        if self.jet_type == 4:
            Z["thetaWing"] = params_dict["thetaWing"]
            Z["thetaCore"] = params_dict["alphaCore"]*params_dict["thetaWing"]
            Z["b"] = params_dict["b"]
        
        # Afterglowpy returns flux in mJys
        tt, nunu = np.meshgrid(self._times_afterglowpy, self.nus)
        mJys = grb.fluxDensity(tt, nunu, **Z)
        return mJys

    def __call__(self, idx):
        param_dict = dict(zip(self.parameter_names, self.X[idx]))
        param_dict.update(self.fixed_parameters)
        mJys = self._call_afterglowpy(param_dict)
        return  idx, np.log(mJys).flatten()



class RunPyblastafterglow:
    def __init__(self, jet_type, times, nus, X, parameter_names, fixed_parameters = {}):
        self.jet_type = jet_type
        jet_conversion = {"-1": "tophat",
                          "0": "gaussian"}
        self.jet_type = jet_conversion[str(self.jet_type)]
        times_seconds = times * days_to_seconds # pyblastafterglow takes seconds as input
        is_log_uniform = np.allclose(np.diff(np.log(times_seconds)), np.log(times_seconds[1])-np.log(times_seconds[0]))
        if is_log_uniform:
            log_dt = np.log(times_seconds[1])-np.log(times_seconds[0])
            self.lc_times = f'array logspace {times_seconds[0]:e} {np.exp(log_dt)*times_seconds[-1]:e} {len(times_seconds)}' # pyblastafterglow only takes this string format
        else:
            dt = times_seconds[1] - times_seconds[0]
            self.lc_times = f'array uniform {times_seconds[0]:e} {times_seconds[-1]+dt:e} {len(times_seconds)}'
        log_dnu = np.log(nus[1]/nus[0])
        self.lc_freqs = f'array logspace {nus[0]:e} {np.exp(log_dnu)*nus[-1]:e} {len(nus)}' # pyblastafterglow only takes this string format
        self.X = X
        self.parameter_names = parameter_names
        self.fixed_parameters = fixed_parameters

    def _call_pyblastafterglow(self,
                         params_dict: dict[str, Float]) -> Float[Array, "n_times"]:
        """
        Run pyblastafterglow to generate a single flux density output, for a given set of parameters. Note that the parameters_dict should contain all the parameters that the model requires.
        The output will be a set of mJys.

        Args:
            Float[Array, "n_times"]: The flux density in mJys at the given times.
        """
        # Define jet structure (analytic; gaussian) -- 3 free parameters 
        struct = dict(
            struct= self.jet_type, # type of the structure tophat or gaussian
            Eiso_c=np.power(10, params_dict["log10_E0"]),  # isotropic equivalent energy of the burst 
            Gamma0c=params_dict["Gamma0"],    # lorentz factor of the core of the jet 
            M0c=-1.,         # mass of the ejecta (if -1 -- inferr from Eiso_c and Gamma0c)
            n_layers_a=21    # resolution of the jet (number of individual blastwaves)
        )
        if "thetaCore" in list(params_dict.keys()):
            struct.update({"theta_c": params_dict['thetaCore']}) # half-opening angle of the winds of the jet
        elif "thetaWing" in list(params_dict.keys()):
            struct.update({"theta_w": params_dict["thetaWing"], "theta_c": param_dict['xCore']*param_dict["thetaWing"]}) # half-opening angle of the winds of the jet
        
        # set model parameters
        P = dict(
                # main model parameters; Uniform ISM -- 2 free parameters
                main=dict(
                    d_l= 3.086e19, # luminocity distance to the source [cm], fix at 10 pc, so that AB magnitude equals absolute magnitude
                    z = 0.0,   # redshift of the source (used in Doppler shifring and EBL table)
                    n_ism=np.power(10, params_dict["log10_n0"]), # ISM density [cm^-3] (assuming uniform)
                    theta_obs= params_dict["inclination_EM"], # observer angle [rad] (from pol to jet axis)  
                    lc_freqs= self.lc_freqs, # frequencies for light curve calculation
                    lc_times= self.lc_times, # times for light curve calculation
                    tb0=1e2, tb1=1e9, ntb=3000, # burster frame time grid boundary, resolution, for the simulation
                ),

                # ejecta parameters; FS only -- 3 free parameters 
                grb=dict(
                    structure=struct, # structure of the ejecta
                    eps_e_fs=np.power(10, params_dict["log10_epsilon_e"]), # microphysics - FS - frac. energy in electrons
                    eps_b_fs=np.power(10, params_dict["log10_epsilon_B"]), # microphysics - FS - frac. energy in magnetic fields
                    p_fs= params_dict["p"], # microphysics - FS - slope of the injection electron spectrum
                    do_lc='yes',      # task - compute light curves
                    rtol_theta = 1e-1,
                    # save_spec='yes' # save comoving spectra 
                    # method_synchrotron_fs = 'Joh06',
                    # method_ne_fs = 'usenprime',
                    # method_ele_fs = 'analytic',
                    # method_comp_mode = 'observFlux'
                )
        )

        pba_run = run_grb(working_dir= os.getcwd() + '/tmp/', # directory to save/load from simulation data
                              P=P,                     # all parameters 
                              run=True,                # run code itself (if False, it will try to load results)
                              path_to_cpp="/home/aya/work/hkoehn/fiesta/PyBlastAfterglowMag/src/pba.out", # absolute path to the C++ executable of the code
                              loglevel="info",         # logging level of the code (info or err)
                              process_skymaps=False    # process unstractured sky maps. Only useed if `do_skymap = yes`
                             )
        mJys = pba_run.GRB.get_lc()
        return mJys

    def __call__(self, idx):
        param_dict = dict(zip(self.parameter_names, self.X[idx]))
        param_dict.update(self.fixed_parameters)
        mJys = self._call_pyblastafterglow(param_dict)
        return  idx, np.log(mJys).flatten()
