"""Method to train the surrogate models"""
import os
from xmlrpc.client import Boolean
import numpy as np
import ast
import h5py
import tqdm
from multiprocessing import Pool, Value

from fiesta.constants import days_to_seconds
import afterglowpy as grb


class AfterglowData:
    def __init__(self,
                 outdir: str,
                 n_training: int, 
                 n_val: int,
                 n_test: int,
                 parameter_distributions: dict = None,
                 jet_type: int = -1,
                 tmin: float = 1.,
                 tmax: float = 1000.,
                 n_times: int = 100,
                 use_log_spacing: bool = True,
                 numin: float = 1e9,
                 numax: float = 2.5e18,
                 n_nu: int = 256,
                 fixed_parameters: dict = {}) -> None:
        
        self.outdir = outdir
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        self.outfile = os.path.join(self.outdir, self.outfile)

        self.n_training = n_training
        self.n_val = n_val
        self.n_test = n_test

        if os.path.exists(self.outfile):
            self._read_file()
        else:
            self.jet_type = jet_type
            if self.jet_type not in [-1,0]:
                raise ValueError(f"Jet type {jet_type} is not supported. Supported jet types are: [-1, 0]")
            self.initialize_times(tmin, tmax, n_times, use_log_spacing) # create time array
            self.initialize_nus(numin, numax, n_nu) # create frequency array
            self.parameter_names = list(parameter_distributions.keys())
            self.parameter_distributions = parameter_distributions
            self._initialize_file() # initialize the h5 file the data is later written to
            self.n_training_exists, self.n_val_exists, self.n_test_exists = 0, 0, 0
        
        print(f"Initialized fiesta.train.AfterglowData \nJet type: {self.jet_type} \nParameters: {self.parameter_names} \nTimes: {self.times[0]} {self.times[-1]} {len(self.times)} \nNus: {self.nus[0]:.3e} {self.nus[-1]:.3e} {len(self.nus)} \nparameter_distributions: {self.parameter_distributions}\nExisting train, val, test: {self.n_training_exists}, {self.n_val_exists}, {self.n_test_exists} \n \n \n")      
        self.fixed_parameters = fixed_parameters

        self.get_raw_data(self.n_training, "train") # create new data and save it to file
        self.get_raw_data(self.n_val, "val")
        self.get_raw_data(self.n_test, "test")

    def initialize_times(self, tmin, tmax, n_times, use_log_spacing: bool = True):
        if use_log_spacing:
            times = np.logspace(np.log10(tmin), np.log10(tmax), num=n_times)
        else:
            times = np.linspace(tmin, tmax, num=n_times)
        self.times = times
    
    def initialize_nus(self, numin: float, numax: float, n_nu: int):
        self.nus = np.logspace(np.log10(numin), np.log10(numax), n_nu)

    def _initialize_file(self,):
        with h5py.File(self.outfile, "w") as f:
            f.create_dataset("times", data = self.times)
            f.create_dataset("nus", data = self.nus)
            f.create_dataset("parameter_names", data = self.parameter_names)
            f.create_dataset("parameter_distributions", data = str(self.parameter_distributions))
            f.create_dataset("jet_type", data = self.jet_type)
            f.create_group("train"); f.create_group("val"); f.create_group("test"); f.create_group("special_train")

    def get_raw_data(self, n: int, group: str):
        if group == "train":
            training = True
        else:
            training = False

        nchunks, rest = divmod(n, self.chunk_size) # create raw data in chunks of chunk_size
        for chunk in tqdm.tqdm([*(nchunks*[self.chunk_size]), rest], desc = f"Calculating {nchunks+1} chunks of {group} data...", leave = True):
            if chunk ==0:
                continue
            X, y = self.create_raw_data(chunk, training)
            X, y = self.fix_nans(X, y)
            self._save_to_file(X, y, group)

    def fix_nans(self,X,y):
        # fixes any nans that remain from create_raw_data
        problematic = np.unique(np.where(np.isnan(y))[0])
        n = len(problematic)
        while n>0:
            if n> 0.1*len(X):
                print(f"Warning: Found many nans for the parameter samples, in total {n} out of {len(X)} samples.")
            X_replacement, y_replacement = self.create_raw_data(n)
            X[problematic] = X_replacement
            y[problematic] = y_replacement
            problematic = np.unique(np.where(np.isnan(y))[0])
            n = len(problematic)
        return X, y
    
    def _read_file(self,):
        with h5py.File(self.outfile, "r") as f:
            self.jet_type = f["jet_type"][()]
            self.times = f["times"][:]
            self.nus = f["nus"][:]
            self.parameter_names = f["parameter_names"][:].astype(str).tolist()
            self.n_training_exists = (f["train"]["X"].shape)[0]
            self.n_val_exists = (f["val"]["X"].shape)[0]
            self.n_test_exists = (f["test"]["X"].shape)[0]
            self.parameter_distributions = ast.literal_eval(f["parameter_distributions"][()].decode('utf-8'))

    def create_raw_data(self, n: int, training: bool = True):
        """
        Create draws X in the parameter space and run the afterglow model on it.
        """
        # Create training data
        X_raw = np.empty((n, len(self.parameter_names)))
       
        if training:
            for j, key in enumerate(self.parameter_names):
                a, b, distribution = self.parameter_distributions[key] # FIXME
                if distribution == "uniform":
                    X_raw[:,j] = np.random.uniform(a, b, size = n)
                elif distribution == "loguniform":
                    X_raw[:,j] = np.exp(np.random.uniform(np.log(a), np.log(b), size = n))
        else:
            for j, key in enumerate(self.parameter_distributions.keys()):
                a, b, _ = self.parameter_distributions[key]
                X_raw[:, j] = np.random.uniform(a, b, size = n)

        # Ensure that epsilon_e +  epsilon_B < 1
        epsilon_e_ind = self.parameter_names.index("log10_epsilon_e")
        epsilon_B_ind = self.parameter_names.index("log10_epsilon_B")
        epsilon_tot = (10**(X_raw[:, epsilon_e_ind]) + 10**(X_raw[:, epsilon_B_ind]))
        mask = epsilon_tot>=1 
        X_raw[mask, epsilon_B_ind] += np.log10(0.99/epsilon_tot[mask])
        X_raw[mask, epsilon_e_ind] += np.log10(0.99/epsilon_tot[mask])
        
        # Ensure that thetaWing is smaller than pi/2
        if self.jet_type !=-1:
            alphaw_ind = self.parameter_names.index("alphaWing")
            thetac_ind = self.parameter_names.index("thetaCore")
            mask = X_raw[:, alphaw_ind]*X_raw[:, thetac_ind] >= np.pi/2
            X_raw[mask, alphaw_ind] = np.pi/2 * 1/X_raw[mask, thetac_ind]

        X, y = self.run_afterglow_model(X_raw)
        return X, y
    
    def create_special_data(self, X_raw, label:str, comment: str = None):
        """Create special training data with pre-specified parameters X. These will be stored in the 'special_train' hdf5 group."""

        # Ensure that epsilon_e +  epsilon_B < 1
        epsilon_e_ind = self.parameter_names.index("log10_epsilon_e")
        epsilon_B_ind = self.parameter_names.index("log10_epsilon_B")
        epsilon_tot = (10**(X_raw[:, epsilon_e_ind]) + 10**(X_raw[:, epsilon_B_ind]))
        mask = epsilon_tot>=1 
        X_raw[mask, epsilon_B_ind] += np.log10(0.99/epsilon_tot[mask])
        X_raw[mask, epsilon_e_ind] += np.log10(0.99/epsilon_tot[mask])
        
        # Ensure that thetaWing is smaller than pi/2
        if self.jet_type !=-1:
            alphaw_ind = self.parameter_names.index("alphaWing")
            thetac_ind = self.parameter_names.index("thetaCore")
            mask = X_raw[:, alphaw_ind]*X_raw[:, thetac_ind] >= np.pi/2
            X_raw[mask, alphaw_ind] = np.pi/2 * 1/X_raw[mask, thetac_ind]
        
        X, y = self.run_afterglow_model(X_raw)
        X, y = self.fix_nans(X,y)
        self._save_to_file(X, y, "special_train", label = label, comment= comment)

    def run_afterglow_model(X):
        raise NotImplementedError

    def _save_to_file(self, X, y, group: str, label: str = None, comment: str = None):
        with h5py.File(self.outfile, "a") as f:
            if "y" in f[group]: # checks if the dataset already exists
                Xset = f[group]["X"]
                Xset.resize(Xset.shape[0]+X.shape[0], axis = 0)
                Xset[-X.shape[0]:] = X

                yset = f[group]["y"]
                yset.resize(yset.shape[0]+y.shape[0], axis = 0)
                yset[-y.shape[0]:] = y
            
            elif label is not None: # or if we have special training data
                if label in f["special_train"]:
                    Xset = f["special_train"][label]["X"]
                    Xset.resize(Xset.shape[0]+X.shape[0], axis = 0)
                    Xset[-X.shape[0]:] = X

                    yset = f[group][label]["y"]
                    yset.resize(yset.shape[0]+y.shape[0], axis = 0)
                    yset[-y.shape[0]:] = y

                else: 
                    f["special_train"].create_group(label)
                    if comment is not None:
                        f["special_train"][label].attrs["comment"] = comment
                    f["special_train"][label].create_dataset("X", data = X, maxshape=(None, len(self.parameter_names)), chunks = (self.chunk_size, len(self.parameter_names)))
                    f["special_train"][label].create_dataset("y", data = y, maxshape=(None, len(self.times)*len(self.nus)), chunks = (self.chunk_size, len(self.times)*len(self.nus)))

            else: # or if we need to create a new data set
                f[group].create_dataset("X", data = X, maxshape=(None, len(self.parameter_names)), chunks = (self.chunk_size, len(self.parameter_names)))
                f[group].create_dataset("y", data = y, maxshape=(None, len(self.times)*len(self.nus)), chunks = (self.chunk_size, len(self.times)*len(self.nus)))

class AfterglowpyData(AfterglowData):

    def __init__(self, n_pool: int, *args, **kwargs):
        self.outfile = "afterglowpy_raw_data.h5"
        self.n_pool = n_pool
        self.chunk_size = 1000
        super().__init__(*args, **kwargs)


    def run_afterglow_model(self, X):
        """Uses multiprocessing to run afterglowpy on the supplied parameters in X."""
        y = np.empty((len(X), len(self.times)*len(self.nus)))
        afgpy = RunAfterglowpy(self.jet_type, self.times, self.nus, X, self.parameter_names, self.fixed_parameters)
        pool = Pool(processes=self.n_pool)
        jobs = [pool.apply_async(func=afgpy, args=(argument,)) for argument in range(len(X))]
        pool.close()
        for Idx, job in enumerate(tqdm.tqdm(jobs, desc = f"Computing {len(X)} afterglowpy calculations.", leave = False)):
            try:
                idx, out = job.get()
                y[idx] = out
            except:
                y[Idx] = np.full(len(self.times)*len(self.nus), np.nan)
        return X, y


class PyblastafterglowData(AfterglowData):

    def __init__(self, path_to_exec: str, rank: int = 0, grb_resolution: int = 12, *args, **kwargs):
        self.outfile = f"pyblastafterglow_raw_data_{rank}.h5"
        self.chunk_size = 10
        self.rank = rank
        self.path_to_exec = path_to_exec
        self.grb_resolution = grb_resolution

        try:
            from PyBlastAfterglowMag.wrappers import run_grb
        except ImportError:
            raise ImportError("PyBlastAfterglowMag is not installed. Please install it from source.")
        super().__init__(*args, **kwargs)


    def run_afterglow_model(self, X):
        """Should be run in parallel with different mpi processes to run pyblastafterglow on the parameters in the array X."""
        y = np.empty((len(X), len(self.times)*len(self.nus)))
        pbag = RunPyblastafterglow(self.jet_type, self.times, self.nus, X, self.parameter_names, self.fixed_parameters, rank=self.rank, path_to_exec = self.path_to_exec, grb_resolution  = self.grb_resolution)
        for j in tqdm.tqdm(range(len(X)), desc = f"Computing {len(X)} pyblastafterglow calculations.", leave = False):
            try:
                idx, out = pbag(j)
                y[idx] = out
            except:
                try: 
                    pbag.n_tb = 3000 # increase blast wave evolution time grid if there is an error
                    idx, out = pbag(j)
                    y[idx] = out
                    pbag.n_tb = 1000
                except:
                    y[j] = np.full(len(self.times)*len(self.nus), np.nan)
                     
        return X, y


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
                         params_dict: dict[str, float]):
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
        
        elif self.jet_type == 0:
             Z["thetaCore"] = params_dict["thetaCore"]
             Z["thetaWing"] = params_dict["thetaCore"]*params_dict["alphaWing"]

        elif self.jet_type == 4:
            Z["thetaCore"] = params_dict["thetaCore"]
            Z["thetaWing"] = params_dict["thetaCore"]*params_dict["alphaWing"]
            Z["b"] = params_dict["b"]
        
        else:
            raise ValueError(f"Provided jet type {self.jet_type} invalid.")
        
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
    def __init__(self, jet_type, times, nus, X, parameter_names, fixed_parameters = {}, rank = 0, path_to_exec = "./pba.out", grb_resolution = 12):
        self.jet_type = jet_type
        jet_conversion = {"-1": "tophat",
                          "0": "gaussian"}
        self.jet_type = jet_conversion[str(self.jet_type)]
        times_seconds = times * days_to_seconds # pyblastafterglow takes seconds as input

        # preparing the pyblastafterglow string argument for time array
        is_log_uniform = np.allclose(np.diff(np.log(times_seconds)), np.log(times_seconds[1])-np.log(times_seconds[0]))
        if is_log_uniform:
            log_dt = np.log(times_seconds[1])-np.log(times_seconds[0])
            self.lc_times = f'array logspace {times_seconds[0]:e} {np.exp(log_dt)*times_seconds[-1]:e} {len(times_seconds)}' # pyblastafterglow only takes this string format
        else:
            dt = times_seconds[1] - times_seconds[0]
            self.lc_times = f'array uniform {times_seconds[0]:e} {times_seconds[-1]+dt:e} {len(times_seconds)}'

        # preparing the pyblastafterglow string argument for frequency array
        log_dnu = np.log(nus[1]/nus[0])
        self.lc_freqs = f'array logspace {nus[0]:e} {np.exp(log_dnu)*nus[-1]:e} {len(nus)}' # pyblastafterglow only takes this string format

        self.X = X
        self.parameter_names = parameter_names
        self.fixed_parameters = fixed_parameters
        self.rank = rank
        self.path_to_exec = path_to_exec
        self.grb_resolution = grb_resolution
        self.n_tb = 1000 # set default blast wave evolution timegrid to 1000

    def _call_pyblastafterglow(self,
                         params_dict: dict[str, float]):
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
            n_layers_a=self.grb_resolution    # resolution of the jet (number of individual blastwaves)
        )

        if self.jet_type == "tophat":
            struct["theta_c"] =  params_dict['thetaCore'] # half-opening angle of the winds of the jet
        
        elif self.jet_type == "gaussian":
            struct["theta_c"] =  params_dict['thetaCore'] # half-opening angle of the winds of the jet
            struct["theta_w"] = params_dict["thetaCore"] * params_dict["alphaWing"]
        
        else:
            raise ValueError(f"Provided jet type {self.jet_type} invalid.")

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
                    tb0=1e1, tb1=1e11, ntb=self.n_tb, # burster frame time grid boundary, resolution, for the simulation
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
        pba_run = run_grb(working_dir= os.getcwd() + f'/tmp_{self.rank}/', # directory to save/load from simulation data
                              P=P,                     # all parameters 
                              run=True,                # run code itself (if False, it will try to load results)
                              path_to_cpp=self.path_to_exec, # absolute path to the C++ executable of the code
                              loglevel="err",         # logging level of the code (info or err)
                              process_skymaps=False    # process unstractured sky maps. Only useed if `do_skymap = yes`
                             )
        mJys = pba_run.GRB.get_lc()
        return mJys

    def __call__(self, idx):
        param_dict = dict(zip(self.parameter_names, self.X[idx]))
        param_dict.update(self.fixed_parameters)
        mJys = self._call_pyblastafterglow(param_dict)
        return  idx, np.log(mJys).flatten()
