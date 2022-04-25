import importlib.util
import json
import logging
import os
import shutil
import tarfile
import tempfile
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Union, Any, Dict, Tuple, BinaryIO
from urllib.parse import urlparse
from zipfile import is_zipfile, ZipFile

import requests
import spacy
import stanza
import yaml
from filelock import FileLock
from spacy.cli.download import download as spacy_download
from tqdm import tqdm

from sapienzanlp.common.logging import get_logger

logger = get_logger(level=logging.DEBUG)


_onnx_available = importlib.util.find_spec("onnx") is not None


def is_onnx_available():
    return _onnx_available


_torch_available = importlib.util.find_spec("torch") is not None


def is_torch_available():
    return _torch_available


# Spacy and Stanza stuff

LOADED_SPACY_MODELS: Dict[Tuple[str, bool, bool, bool, bool], spacy.Language] = {}


def load_spacy(
    language: str,
    pos_tags: bool = False,
    lemma: bool = False,
    parse: bool = False,
    split_on_spaces: bool = False,
) -> spacy.Language:
    """
    Download and load spacy model.

    Args:
        language:
        pos_tags:
        lemma:
        parse:
        split_on_spaces:

    Returns:
        spacy.Language: The spacy tokenizer loaded.
    """
    exclude = ["vectors", "textcat", "ner"]
    if not pos_tags:
        exclude.append("tagger")
    if not lemma:
        exclude.append("lemmatizer")
    if not parse:
        exclude.append("parser")

    # check if the model is already loaded
    # if so, there is no need to reload it
    spacy_params = (language, pos_tags, lemma, parse, split_on_spaces)
    if spacy_params not in LOADED_SPACY_MODELS:
        try:
            spacy_tagger = spacy.load(language, exclude=exclude)
        except OSError:
            logger.warning(f"Spacy model '{language}' not found. Downloading and installing.")
            spacy_download(language)
            spacy_tagger = spacy.load(language, exclude=exclude)

        # if everything is disabled, return only the tokenizer
        # for faster tokenization
        # if len(exclude) >= 6:
        #     spacy_tagger = spacy_tagger.tokenizer
        LOADED_SPACY_MODELS[spacy_params] = spacy_tagger

    return LOADED_SPACY_MODELS[spacy_params]


LOADED_STANZA_MODELS: Dict[Tuple[str, str, bool, bool], stanza.Pipeline] = {}


def load_stanza(
    language: str = "en",
    pos_tags: bool = False,
    lemma: bool = False,
    parse: bool = False,
    tokenize_pretokenized: bool = False,
    use_gpu: bool = False,
) -> stanza.Pipeline:
    """
    Download and load stanza model.

    Args:
        language:
        pos_tags:
        lemma:
        parse:
        tokenize_pretokenized:
        use_gpu:

    Returns:
        stanza.Pipeline: The stanza tokenizer loaded.

    """
    processors = ["tokenize"]
    if pos_tags:
        processors.append("pos")
    if lemma:
        processors.append("lemma")
    if parse:
        processors.append("depparse")
    processors = ",".join(processors)

    # check if the model is already loaded
    # if so, there is no need to reload it
    stanza_params = (language, processors, tokenize_pretokenized, use_gpu)
    if stanza_params not in LOADED_STANZA_MODELS:
        try:
            stanza_tagger = stanza.Pipeline(
                language,
                processors=processors,
                tokenize_pretokenized=tokenize_pretokenized,
                tokenize_no_ssplit=True,
                use_gpu=use_gpu,
            )
        except OSError:
            logger.info(f"Stanza model '{language}' not found. Downloading and installing.")
            stanza.download(language)
            stanza_tagger = stanza.Pipeline(
                language,
                processors=processors,
                tokenize_pretokenized=tokenize_pretokenized,
                tokenize_no_ssplit=True,
                use_gpu=use_gpu,
            )
        LOADED_STANZA_MODELS[stanza_params] = stanza_tagger

    return LOADED_STANZA_MODELS[stanza_params]

# file I/O stuff


def load_yaml(path: Union[str, Path]):
    """
    Load a yaml file provided in input.
    Args:
        path: path to the yaml file.

    Returns:
        The yaml file parsed.
    """
    with open(path, encoding="utf8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def dump_yaml(document: Any, path: Union[str, Path]):
    """
    Dump input to yaml file.
    Args:
        document: thing to dump
        path: path to the yaml file.

    Returns:

    """
    with open(path, "w", encoding="utf8") as outfile:
        yaml.dump(document, outfile, default_flow_style=False)


def load_json(path: Union[str, Path]):
    """
    Load a yaml file provided in input.
    Args:
        path: path to the json file.

    Returns:
        The yaml file parsed.
    """
    with open(path, encoding="utf8") as f:
        return json.load(f)


def dump_json(document: Any, path: Union[str, Path]):
    """
    Dump input to json file.
    Args:
        document: thing to dump
        path: path to the yaml file.

    Returns:

    """
    with open(path, "w", encoding="utf8") as outfile:
        json.dump(document, outfile, indent=2)


# remote url
ENDPOINT_URL = "http://localhost:8000/models"
# cache dir
SAPIENZANLP_CACHE_DIR = os.getenv("SAPIENZANLP_CACHE_DIR", Path.home() / ".sapienzanlp")
SAPIENZANLP_MODEL_DIR = os.getenv("SAPIENZANLP_MODEL_DIR", SAPIENZANLP_CACHE_DIR / "models")
# name constants
WEIGHTS_NAME = "weights.pt"
ONNX_WEIGHTS_NAME = "weights.onnx"
CONFIG_NAME = "config.json"
LABELS_NAME = "labels.json"
EXTRAS_NAME = "extras.json"


def sapienzanlp_model_urls(model_id: str) -> str:
    """
    Returns the URL for a possible SapienzaNLP valid model.

    Args:
        model_id (:obj:`str`):
            A SapienzaNLP model id.

    Returns:
        :obj:`str`: The url for the model id.
    """
    return f"{ENDPOINT_URL}/{model_id}.zip"


def is_remote_url(url_or_filename: Union[str, Path]):
    """
    Returns :obj:`True` if the input path is an url.

    Args:
        url_or_filename (:obj:`str`, :obj:`Path`):
            path to check.

    Returns:
        :obj:`bool`: :obj:`True` if the input path is an url, :obj:`False` otherwise.

    """
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def check_exists_dir(path: Union[str, Path]):
    """
    Create dir in case it does not exist.

    Args:
        path (:obj:`str`, :obj:`Path`): path to check.

    """
    Path(path).mkdir(parents=True, exist_ok=True)


def file_exists(path: Union[str, Path]) -> bool:
    """
    Check if the file at :obj:`path` exists.

    Args:
        path (:obj:`str`, :obj:`Path`):
            Path to check.

    Returns:
        :obj:`bool`: :obj:`True` if the file exists.
    """
    return Path(path).exists()


def dir_exists(path: Union[str, Path]) -> bool:
    """
    Check if the directory at :obj:`path` exists.

    Args:
        path (:obj:`str`, :obj:`Path`):
            Path to check.

    Returns:
        :obj:`bool`: :obj:`True` if the directory exists.
    """
    return Path(path).is_dir()


def url_to_filename(resource: str, etag: str = None) -> str:
    """
    Convert a `resource` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the resources's, delimited
    by a period.
    """
    resource_bytes = resource.encode("utf-8")
    resource_hash = sha256(resource_bytes)
    filename = resource_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    return filename


def filename_to_url(filename: str, cache_dir: Union[str, Path] = None) -> Tuple[str, str]:
    """
    Return the url and etag (which may be `None`) stored for `filename`.
    Raise `FileNotFoundError` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = SAPIENZANLP_CACHE_DIR

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise FileNotFoundError("file {} not found".format(cache_path))

    meta_path = cache_path + ".json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError("file {} not found".format(meta_path))

    with open(meta_path) as meta_file:
        metadata = json.load(meta_file)
    url = metadata["url"]
    etag = metadata["etag"]

    return url, etag


def download_resource(
    url: str,
    temp_file: BinaryIO,
):
    """
    Download remote file.
    """
    r = requests.get(url, stream=True)
    r.raise_for_status()
    content_length = r.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = tqdm(
        unit="B",
        unit_scale=True,
        total=total,
        desc="Downloading",
        disable=logger.level in [logging.NOTSET],
    )
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def download_and_cache(
    url: Union[str, Path],
    cache_dir: Union[str, Path] = None,
    force_download: bool = False,
):
    if cache_dir is None:
        cache_dir = SAPIENZANLP_CACHE_DIR
    if isinstance(url, Path):
        url = str(url)

    # check if cache dir exists
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # TODO etag for checking if a model is updated.
    # etag = None
    # try:
    #     r = requests.head(url, allow_redirects=False, timeout=10)
    #     r.raise_for_status()
    #     etag = r.headers.get("X-Linked-Etag") or r.headers.get("ETag")
    #     # We favor a custom header indicating the etag of the linked resource, and
    #     # we fallback to the regular etag header.
    #     # If we don't have any of those, raise an error.
    #     if etag is None:
    #         raise OSError(
    #             "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility."
    #         )
    #     # In case of a redirect,
    #     # save an extra redirect on the request.get call,
    #     # and ensure we download the exact atomic version even if it changed
    #     # between the HEAD and the GET (unlikely, but hey).
    #     if 300 <= r.status_code <= 399:
    #         url = r.headers["Location"]
    # except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
    #     # Actually raise for those subclasses of ConnectionError
    #     raise
    # except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
    #     # Otherwise, our Internet connection is down.
    #     # etag is None
    #     pass

    # get filename from the url
    filename = url_to_filename(url)
    # get cache path to put the file
    cache_path = cache_dir / filename

    # the file is already here, return it
    if file_exists(cache_path) and not force_download:
        logger.info(f"{url} found in cache, set `force_download=True` to force the download")
        return cache_path

    cache_path = str(cache_path)
    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):

        # If the download just completed while the lock was activated.
        if file_exists(cache_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return cache_path

        temp_file_manager = partial(
            tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False
        )

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info(
                f"{url} not found in cache or `force_download` set to `True`, downloading to {temp_file.name}"
            )
            download_resource(url, temp_file)

        logger.info(f"storing {url} in cache at {cache_path}")
        os.replace(temp_file.name, cache_path)

        # NamedTemporaryFile creates a file with hardwired 0600 perms (ignoring umask), so fixing it.
        umask = os.umask(0o666)
        os.umask(umask)
        os.chmod(cache_path, 0o666 & ~umask)

        logger.info(f"creating metadata file for {cache_path}")
        meta = {"url": url}  # , "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)

    return cache_path


def from_cache(
    url_or_filename: Union[str, Path],
    cache_dir: Union[str, Path] = None,
    force_download: bool = False,
) -> Path:
    """

    Args:
        url_or_filename:
        cache_dir:
        force_download:

    Returns:

    """
    if cache_dir is None:
        cache_dir = SAPIENZANLP_CACHE_DIR

    if is_remote_url(url_or_filename):
        # URL, so get it from the cache (downloading if necessary)
        output_path = download_and_cache(
            url_or_filename,
            cache_dir=cache_dir,
            force_download=force_download,
        )
    elif file_exists(url_or_filename):
        logger.info(f"{url_or_filename} is a local path or file")
        # File, and it exists.
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError(f"file {url_or_filename} not found")
    else:
        # Something unknown
        raise ValueError(f"unable to parse {url_or_filename} as a URL or as a local path")

    if dir_exists(output_path) or (
        not is_zipfile(output_path) and not tarfile.is_tarfile(output_path)
    ):
        return Path(output_path)

    # Path where we extract compressed archives
    # for now it will extract it in the same folder
    # maybe implement extraction in the sapienzanlp folder
    # when using local archive path?
    logger.info("Extracting compressed archive")
    output_dir, output_file = os.path.split(output_path)
    output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
    output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

    # already extracted, do not extract
    if (
        os.path.isdir(output_path_extracted)
        and os.listdir(output_path_extracted)
        and not force_download
    ):
        return Path(output_path_extracted)

    # Prevent parallel extractions
    lock_path = output_path + ".lock"
    with FileLock(lock_path):
        shutil.rmtree(output_path_extracted, ignore_errors=True)
        os.makedirs(output_path_extracted)
        if is_zipfile(output_path):
            with ZipFile(output_path, "r") as zip_file:
                zip_file.extractall(output_path_extracted)
                zip_file.close()
        elif tarfile.is_tarfile(output_path):
            tar_file = tarfile.open(output_path)
            tar_file.extractall(output_path_extracted)
            tar_file.close()
        else:
            raise EnvironmentError(f"Archive format of {output_path} could not be identified")

    # remove lock file, is it safe?
    os.remove(lock_path)

    return Path(output_path_extracted)
