import os
from pathlib import Path
from shutil import copy2, rmtree

from fiesta.logging import logger

from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.utils import HfHubHTTPError

HF_REPO_ID = "nuclear-multimessenger-astronomy/fiesta-surrogates"
HF_REVISION = "main"

# Environment variable that, if set, overrides where built-in surrogates are looked up/downloaded
# to. Its value must be (or will become) a directory with "KN" and "GRB" subdirectories, 
# mirroring the layout of the packaged fiesta.surrogates directory.
FIESTA_BUILT_IN_SURROGATE_DIR = "FIESTA_BUILT_IN_SURROGATE_DIR"

TRANSIENT_TYPES = ("KN", "GRB")

###########################
### BUILT-IN SURROGATES ###
###########################


def get_surrogate_dir() -> Path:
    """
    Resolve the base directory holding the built-in surrogates (with "KN" and "GRB"
    subdirectories) -- i.e. the directory that is scanned for already-present
    surrogates and that new downloads are placed into.

    If the ``FIESTA_BUILT_IN_SURROGATE_DIR`` environment variable is set, that
    directory is used instead of the default (and its ``KN``/``GRB`` subdirectories
    are created if they don't exist yet). Otherwise this defaults to the
    ``fiesta.surrogates`` package directory that ships with fiesta, which already
    contains those subdirectories.
    """
    env_dir = os.environ.get(FIESTA_BUILT_IN_SURROGATE_DIR)
    if env_dir:
        surrogate_dir = Path(env_dir).expanduser().resolve()
        for transient in TRANSIENT_TYPES:
            (surrogate_dir / transient).mkdir(parents=True, exist_ok=True)
    else:
        surrogate_dir = Path(__file__).resolve().parent

    return surrogate_dir


def built_in_surrogates():
    surrogate_dir = get_surrogate_dir()

    for transient_dir in sorted(surrogate_dir.iterdir()):
        if not transient_dir.is_dir():
            continue
        if not transient_dir.parts[-1] in TRANSIENT_TYPES:
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


#################################
### DOWNLOAD FROM HUGGINGFACE ###
#################################

def download_surrogate(
        name: str,
        directory: str | None = None,
    ) -> tuple[bool, str | None]:
    """
    Downloads a surrogate from the fiesta hugging-face repository.

    Args:
        name (str): Which surrogate to download. Available downloads can be checked with ``print_downloadable_surrogates``.
        directory (str | None): Where to download the surrogate.
        Defaults to ``None``, in which case the surrogate will be downloaded to the built-in surrogate directory (see ``get_surrogate_dir``) from where it can be loaded automatically.

    Returns:
        download_ok (bool): Whether the download was successful.
        surrogate_dir (str): Location where the surrogate was downloaded to.
    """
    
    if name.endswith("_lc"):
        raise ValueError("Light curve models are not supported for automatic download at the moment. Please download manually from Hugging Face.")

    logger.info(f"Attempting to download {name} from Hugging Face ({HF_REPO_ID}).")

    download_ok = False
    for transient in TRANSIENT_TYPES:

        try:
            metadata_path = f"{transient}/{name}/{name}_metadata.pkl"
            downloaded_metadata = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    revision=HF_REVISION,
                    filename=metadata_path,
                )

            if directory is None:
                download_dir = get_surrogate_dir()
                Path(download_dir / f"{transient}/{name}").mkdir(parents=True, exist_ok=True)
            else:
                download_dir = Path(directory)
                Path(download_dir).mkdir(parents=True, exist_ok=True)

            if directory is not None:
                metadata_path = f"{name}_metadata.pkl"
            if (download_dir / metadata_path).exists():
                logger.warning(f"Surrogate metadata for {name} already present in {download_dir / metadata_path}. Will be overwritten through download.")

            copy2(downloaded_metadata, download_dir / metadata_path)
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

    model_path = f"{transient}/{name}/{name}.pkl"
    try:
        downloaded_pkl = hf_hub_download(
            repo_id=HF_REPO_ID,
            revision=HF_REVISION,
            filename=model_path,
        )

        if directory is not None:
            model_path = f"{name}.pkl"

        copy2(downloaded_pkl, download_dir / model_path)

    except EntryNotFoundError:
        logger.warning(f"Model file not found on Hugging Face: {model_path}")
        return False, None
    
    except HfHubHTTPError:
        logger.exception(f"Hugging Face model download failed for transient={transient}, model={name}.")
        raise

    if directory is None:
        download_dir = download_dir / transient / name
    logger.info(f"Successfully downloaded {name} to {download_dir}.")

    return download_ok, download_dir

def download_recommended_surrogates():

    download_surrogate("Bu2026_MLP")
    download_surrogate("afgpy_gaussian_CVAE")
    download_surrogate("pbag_gaussian_CVAE")

def print_downloadable_surrogates():

    files = HfApi().list_repo_files(
        repo_id=HF_REPO_ID,
        revision=HF_REVISION,
    )

    available = {transient: set() for transient in TRANSIENT_TYPES}

    for path in files:
        # Expected structure:
        # {transient}/<name>/<name>_metadata.pkl
        parts = path.split("/")
        if len(parts) == 3 and parts[2].endswith("_metadata.pkl") and not parts[1].startswith("."):
            transient = parts[0]
            name = parts[1]
            available.setdefault(transient, set()).add(name)

    if not any(available.values()):
        print("No surrogate models found.")
        return

    logger.info(f"Downloadable surrogates for fiesta are:")
    for transient, models in available.items():
        logger.info(f"\t ==================== ")
        logger.info(f"\t \t-{transient}-")
        for name in models:
            logger.info(f"\t {name}")
        logger.info(f"\t ==================== \n \n")
