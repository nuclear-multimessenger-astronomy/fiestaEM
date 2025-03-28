import os

import numpy as np
import yaml

from fiesta.logging import logger



def setup_systematic(likelihood, prior, systematics_file: str=None):

    if systematics_file is None:
        if "em_syserr" in prior.naming:
            likelihood._setup_sys_uncertainty(sys_uncertainty_sampling=True)
            logger.info(f"Likelihood is using freely sampled systematic uncertainty as specified in the prior.")
        else:
            logger.info(f"Likelihood is using fixed systematic uncertainty {self.likelihood.error_budget}.")
    
    else:
        if not os.path.exists(systematics_file):
            raise OSError(f"Provided systematics file {systematics_file} could not be found.")
        
        filter_uncertainty_dict, additional_priors = process_file(systematics_file, likelihood.filters)
    

    return likelihood, prior


def process_file(systematic_file, filters):
    
    yaml_dict = yaml.safe_load(systematic_file)

    raise ValueError(f"To be continued")



        
""""

def sys_uncerta_interp(filter, t_det, mag_err):

    t_nodes, 

    sys_uncertainty = jnp.interp(t_det, t_nodes, sys_uncertainties)

    sigma = jnp.sqrt(mag_err**2 + sys_uncertainty**2)

    return sigma

jax.tree.map(sys_uncerta_interp, self.filters, self.t_det, self.mag_err)

"""