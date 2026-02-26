import os
from pathlib import Path

from fiesta.logging import logger

import requests

###########################
### BUILT-IN SURROGATES ###
###########################


def built_in_surrogates():
    surrogate_dir = Path(__file__).resolve().parent

    for transient_dir in sorted(surrogate_dir.iterdir()):
        if not transient_dir.is_dir():
            continue
        transient_type = transient_dir.name

        for model_dir in sorted(transient_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name

            if not model_name.startswith("_"):
                yield model_name, model_dir, transient_type

def print_built_in_surrogates():
    logger.info(f"Available built-in surrogates in fiesta are:")
    for model_name, _, transient_type in built_in_surrogates():
        logger.info(f"\t {model_name} ({transient_type})")



def download_surrogate(name):

    if name.endswith("_lc"):
        raise ValueError(f"Light curve models are not supported for download at the moment. Please download manually from github.")

    working_dir = Path(__file__).resolve().parent
    base_url = "https://raw.githubusercontent.com/nuclear-multimessenger-astronomy/fiestaEM/main/surrogates/"

    logger.info(f"Attempting to download {name} from {base_url}.")

    download_ok = False
    for transient in ["KN", "GRB"]:
        url_metadata = f"{base_url}{transient}/{name}/model/{name}_metadata.pkl"
        metadata = requests.get(url_metadata, timeout=10)
    
        if metadata.status_code == 200:
            download_ok = True
            logger.info(f"Found metadata in {url_metadata}. Downloading ...")
            break
    
    if not download_ok:
        return download_ok, None
    
    surrogate_dir = working_dir / transient / name / "model"
    surrogate_dir.mkdir(parents=True, exist_ok=True)
    
    # save metadata
    with open(surrogate_dir / f"{name}_metadata.pkl", "wb") as f:
        f.write(metadata.content)

    url_model = base_url + f"{transient}/{name}/model/{name}.pkl"
    model_pkl = requests.get(url_model, stream=True)
    model_pkl.raise_for_status()
    
    # save model
    with open(surrogate_dir / f"{name}.pkl", "wb") as f:
        f.write(model_pkl.content)
    
    logger.info(f"Download finished.")
    return download_ok, surrogate_dir
