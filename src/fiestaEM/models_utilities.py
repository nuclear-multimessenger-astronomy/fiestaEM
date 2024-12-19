"""Utilities regarding (surrogate) models."""
import numpy as np
import re

####################
### BULLA MODELS ###
####################

### Bu2019lm ###

BU2019LM_PARAMETER_NAMES = ["log10_mej_dyn", 
                            "log10_mej_wind", 
                            "KNphi", 
                            "KNtheta"]

def extract_Bu2019lm_parameters(filename: str) -> dict:
    """
    Extract the parameter values from the filename of a Bulla file

    Args:
        filename (str): Bu2019lm filename, e.g. `./nph1.0e+06_mejdyn0.005_mejwind0.130_phi45_theta84.26_dMpc0.dat`

    Returns:
        np.array: Array with the parameter values for this filename
    """

    # Skip nph entry
    parameters_idx = [1, 2, 3, 4]
    rr = [
        float(x)
        for x in re.findall(
            r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", filename.replace("nph1.0e+06_", "")
        )
    ]

    # Best to interpolate mass in log10 space
    rr[1] = np.log10(rr[1])
    rr[2] = np.log10(rr[2])

    parameter_values = np.array([rr[idx] for idx in parameters_idx])

    return parameter_values


BULLA_PARAMETER_NAMES = {"Bu2019lm": BU2019LM_PARAMETER_NAMES}
SUPPORTED_BULLA_MODELS = list(BULLA_PARAMETER_NAMES.keys())

EXTRACT_PARAMETERS_FUNCTIONS = {"Bu2019lm": extract_Bu2019lm_parameters}

### All

# TODO: update this dict to include any other models
PARAMETER_NAMES = BULLA_PARAMETER_NAMES
SUPPORTED_MODELS = list(PARAMETER_NAMES.keys())