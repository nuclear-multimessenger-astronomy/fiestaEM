import os
from pathlib import Path
from shutil import copy2, rmtree

from fiesta.logging import logger

from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.utils import HfHubHTTPError

HF_REPO_ID = "nuclear-multimessenger-astronomy/fiesta-surrogates"
HF_REVISION = "main"

###########################
### BUILT-IN SURROGATES ###
###########################


def built_in_surrogates():
    surrogate_dir = Path(__file__).resolve().parent

    for transient_dir in sorted(surrogate_dir.iterdir()):
        if not transient_dir.is_dir():
            continue
        if ".cache" in transient_dir.parts:
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
        raise ValueError("Light curve models are not supported for download at the moment. Please download manually from Hugging Face.")

    working_dir = Path(__file__).resolve().parent

    logger.info(f"Attempting to download {name} from Hugging Face ({HF_REPO_ID}).")

    download_ok = False
    for transient in ["KN", "GRB"]:
        try:
            metadata_path = f"{transient}/{name}/model/{name}_metadata.pkl"
            downloaded_file = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    revision=HF_REVISION,
                    filename=metadata_path,
                )

            Path(working_dir / f"{transient}/{name}/model").mkdir(parents=True, exist_ok=True)
            if (working_dir / metadata_path).exists():
                logger.warning(f"Surrogate metadata for {name} already present in {working_dir / metadata_path}. Will be overwritten through download.")
            copy2(downloaded_file, working_dir / metadata_path)
            download_ok = True
            logger.info(f"Found {metadata_path}. Downloading model ...")
            break
        except EntryNotFoundError:
            continue
        except HfHubHTTPError:
            logger.exception(f"Hugging Face lookup failed for transient={transient}, model={name}.")
            raise

    if not download_ok:
        return download_ok, None

    model_path = f"{transient}/{name}/model/{name}.pkl"
    try:
        hf_hub_download(
            repo_id=HF_REPO_ID,
            revision=HF_REVISION,
            filename=model_path,
            local_dir=working_dir,
        )
    except EntryNotFoundError:
        logger.warning(f"Model file not found on Hugging Face: {model_path}")
        return False, None
    except HfHubHTTPError:
        logger.exception(f"Hugging Face model download failed for transient={transient}, model={name}.")
        raise

    surrogate_dir = working_dir / transient / name
    logger.info(f"Download finished.")
    
    rmtree(working_dir / ".cache")


    return download_ok, surrogate_dir

def download_recommended_surrogates():

    download_surrogate("Bu2026_MLP")
    download_surrogate("afgpy_gaussian_CVAE")
    download_surrogate("pbag_gaussian_CVAE")

def print_downloadable_surrogates():

    files = HfApi().list_repo_files(
        repo_id=HF_REPO_ID,
        revision=HF_REVISION,
    )

    available = {"KN": set(), "GRB": set()}

    for path in files:
        # Expected structure:
        # {transient}/<name>/model/<name>_metadata.pkl
        parts = path.split("/")
        if len(parts) == 4 and parts[2] == "model" and parts[3].endswith("_metadata.pkl"):
            transient = parts[0]
            name = parts[1]
            available[transient].add(name)

    if not available:
        print("No surrogate models found.")
        return

    logger.info(f"Downloadable surrogates for fiesta are:")
    for transient, models in available.items():
        logger.info(f"\t ==================== ")
        logger.info(f"\t \t-{transient}-")
        for name in models:
            logger.info(f"\t {name}")
        logger.info(f"\t ==================== \n \n")
