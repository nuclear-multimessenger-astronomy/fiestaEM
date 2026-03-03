import os
from pathlib import Path

from fiesta.logging import logger

from huggingface_hub import hf_hub_download
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
            hf_hub_download(
                repo_id=HF_REPO_ID,
                revision=HF_REVISION,
                filename=metadata_path,
                local_dir=working_dir,
            )
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
    hf_hub_download(
        repo_id=HF_REPO_ID,
        revision=HF_REVISION,
        filename=model_path,
        local_dir=working_dir,
    )

    surrogate_dir = working_dir / transient / name
    logger.info(f"Download finished.")
    return download_ok, surrogate_dir
