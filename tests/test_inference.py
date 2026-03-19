from os.path import dirname, join
from pathlib import Path
import shutil
from multiprocessing import Process

import numpy as np
import jax
import jax.numpy as jnp


from fiesta.inference.prior import Uniform, Constraint, Normal, UniformSourceFrame, ConstrainedPrior, Sine

from fiesta.inference.lightcurve_model import FluxModel, CombinedSurrogate
from fiesta.inference.injection import InjectionSurrogate
from fiesta.inference.systematic import process_file
from fiesta.inference.likelihood import EMLikelihood
from fiesta.utils import load_event_data
from fiesta.inference.fiesta import Fiesta

FILTERS_KN = ["besselli", "bessellv"]
FILTERS_GRB = ["besselli", "radio-3GHz"]

working_dir = Path(dirname(__file__))

def test_injection():

    model_afg = FluxModel(name="pbag_gaussian_CVAE",
                          filters = FILTERS_GRB)
    
    model_kn = FluxModel(name="Bu2026_MLP",
                         filters = FILTERS_KN)
    
    model = CombinedSurrogate(models=[model_kn, model_afg],
                              sample_times=jnp.geomspace(0.3, 1000, 200))


    injection = InjectionSurrogate(model=model,
                                    filters=["besselli", "radio-3GHz"],
                                    trigger_time=69807,
                                    tmin=0.5,
                                    tmax=10,
                                    N_datapoints=20,
                                    error_budget=0.1,
                                    nondetections=True,
                                    detection_limit={"besselli": 25.0, "radio-3GHz": 26.0})
    
    injection.create_injection(dict(inclination_EM=np.pi/18,
                                    log10_E0=52.,
                                    thetaCore=5*np.pi/180,
                                    alphaWing=2.,
                                    log10_n0=-2.,
                                    p=2.15,
                                    log10_epsilon_e=-1,
                                    log10_epsilon_B=-3.,
                                    Gamma0=500.,
                                    log10_mej_dyn=-2.,
                                    v_ej_dyn = 0.2,
                                    Ye_dyn =0.2,
                                    log10_mej_wind=-1.5,
                                    v_ej_wind=0.08,
                                    Ye_wind=0.35,
                                    luminosity_distance=40.0,
                                    redshift=0.)
                                )
    
    injection.write_to_file(join(working_dir, "injection_tmp.dat"))
    Path(join(working_dir, "injection_tmp.dat")).unlink()
    Path(join(working_dir, "param_dict.dat")).unlink()


def test_systematic():
    data = load_event_data(join(working_dir, "injection_KN_GRB_afterglow.dat"))
        
    model = FluxModel(name="pbag_gaussian_CVAE",
                         filters = FILTERS_GRB)

    likelihood = EMLikelihood(model,
                              data,
                              tmin=0.3,
                              tmax=10.,
                              trigger_time=69807,
                              fixed_params={})
    
    
    likelihood._setup_sys_uncertainty_fixed(1.0)

    likelihood._setup_sys_uncertainty_free()


    sys_params_per_filter, time_nodes_per_filter, _ = process_file(join(working_dir, "test_systematics.yaml"), FILTERS_GRB)
    likelihood._setup_sys_uncertainty_from_file(sys_params_per_filter, time_nodes_per_filter)


#################
# test sampling #
#################

def run_with_timeout(func, timeout, *args, **kwargs):

    p = Process(target=func, args=args, kwargs=kwargs)
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return True
    return False

def setup_fiesta_sampling(sampler: str, **kwargs):

    data = load_event_data(join(working_dir, "injection_KN_GRB_afterglow.dat"))
        
    model = FluxModel(name="pbag_gaussian_CVAE",
                         filters = FILTERS_GRB)

    likelihood = EMLikelihood(model,
                              data,
                              tmin=0.3,
                              tmax=10.,
                              trigger_time=69807,
                              fixed_params={})
    
    #########
    # PRIOR #
    #########

    KN_prior = [
                Uniform(xmin=-4.0, xmax=-1.3, naming=["log10_mej_dyn"]),
                Uniform(xmin=0.12, xmax=0.35, naming=["v_ej_dyn"]),
                Uniform(xmin=0.15, xmax=0.35, naming=["Ye_dyn"]),
                Uniform(xmin=-4., xmax=-0.56, naming=["log10_mej_wind"]),
                Uniform(xmin=0.05, xmax=0.15, naming=["v_ej_wind"]),
                Uniform(xmin=0.2, xmax=0.4, naming=["Ye_wind"])
    ]
    
    GRB_prior = [
                 Uniform(xmin=47.0, xmax=57.0, naming=['log10_E0']), 
                 Uniform(xmin=0.01, xmax=np.pi/5, naming=['thetaCore']),
                 Uniform(xmin = 0.2, xmax=3.5, naming= ["alphaWing"]),
                 Constraint(xmin = 0, xmax=np.pi/2, naming = ["thetaWing"]),
                 Uniform(xmin=-6.0, xmax=2.0, naming=['log10_n0']),
                 Uniform(xmin=2.01, xmax=3.0, naming=['p']),
                 Uniform(xmin=-4.0, xmax=0.0, naming=['log10_epsilon_e']),
                 Uniform(xmin=-8.0, xmax=0.0, naming=['log10_epsilon_B']),
                 Uniform(xmin=100., xmax=1000., naming=['Gamma0']),
                 Constraint(xmin = 0., xmax=1., naming=["epsilon_tot"])
    ]

    obs_prior = [
        Sine(xmin=0.0, xmax=np.pi/2, naming=['inclination_EM']),
        UniformSourceFrame(dmin=10, dmax=100., naming=['luminosity_distance']),
        Normal(mu=0.0098, sigma=0.0008, naming=['redshift'])
    ]
    
    def conversion_function(sample):
        converted_sample = sample
        converted_sample["thetaWing"] = converted_sample["thetaCore"] * converted_sample["alphaWing"]
        converted_sample["epsilon_tot"] = 10**(converted_sample["log10_epsilon_B"]) + 10**(converted_sample["log10_epsilon_e"]) 
        return converted_sample
    
    prior_list = [*GRB_prior,
                  *KN_prior,
                  *obs_prior]
    
    prior = ConstrainedPrior(prior_list, conversion_function)
    
    ################
    # LIKELIHOOD & #
    # SAMPLING     #
    ################
      
      
    detection_limit = None
    likelihood = EMLikelihood(model,
                              data,
                              tmin=0.3,
                              tmax=100.,
                              trigger_time=69807,
                              detection_limit = detection_limit,
                              fixed_params={})
    
    
    # Save for postprocessing
    outdir = f"./outdir_joint/"
    
    fiesta = Fiesta(likelihood,
                    prior,
                    systematics_file=join(working_dir, "test_systematics.yaml"),
                    sampler=sampler,
                    **kwargs)
    
    return fiesta
    

def test_flowmc():

    fiesta = setup_fiesta_sampling("flowmc",
                                   n_chains=10,
                                   n_local_steps=2,
                                   n_global_steps=2,
                                   n_training_loops=2,
                                   n_production_loops=2,
                                   n_max_examples=2)

    run_with_timeout(fiesta.sample, 30, jax.random.key(0))