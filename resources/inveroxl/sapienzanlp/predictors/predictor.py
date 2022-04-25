from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union, List

from sapienzanlp.common.from_config import FromConfig
from sapienzanlp.common.logging import get_logger
from sapienzanlp.common.utils import (
    is_remote_url,
    ONNX_WEIGHTS_NAME,
    WEIGHTS_NAME,
    from_cache,
    sapienzanlp_model_urls,
    EXTRAS_NAME,
)
from sapienzanlp.data.model_io.model_inputs import ModelInputs
from sapienzanlp.data.model_io.sentence import Sentence
from sapienzanlp.data.model_io.word import Word
from sapienzanlp.models.model import Model

logger = get_logger(level=logging.DEBUG)


class Predictor(FromConfig):
    """
    Base predictor class. Every predictor should extend this class.

    Args:
        model (:obj:`~sapienzanlp.models.Model`):
            The :obj:`~sapienzanlp.models.Model` that this predictor will use to produce the output.
        language (:obj:`str`):
            Language of the text in input to the predictor. It is used by the
            :obj:`~sapienzanlp.preprocessing.Tokenizer` to preprocess the text. It is ignored if the text is
            already preprocessed.
    """

    def __init__(self, model: Model, language: str = None):
        self.model = model
        # maybe not every model has device, for example ONNX models
        # default to `cpu` even if it isn't used
        self.device = model.device if hasattr(model, "device") else "cpu"
        self.language = language
        self.tokenizer_kwargs = {"language": language}
        self.tokenizer = None

    def __call__(self, *args, **kwargs):
        """
        Every predictor should expose its inference logic here.
        """
        raise NotImplementedError

    def prepare_input_for_model(
        self, text: Union[List[List[str]], List[str], str], *args, **kwargs
    ) -> Any:
        """
        Prepare the input for the model. Every predictor should implement this.

        Args:
            text (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                Text to tag. It can be a single string, a batch of string and pre-tokenized strings.

        Returns:

        """
        raise NotImplementedError

    def make_output(self, texts: List[List[Word]], model_outputs: Any) -> List[Sentence]:
        """

        Args:
            texts (:obj:`List[List[Word]]`):
                Text passed in input.
            model_outputs (:obj:`Any`):
                Output of the model

        Returns:
            (:obj:`List[Sentence]`): model output processed.
        """
        raise NotImplementedError

    def decode(self, model_inputs: ModelInputs, model_outputs: Any) -> Any:
        """
        Decoding function that every predictor should implement

        Args:
            model_inputs (:obj:`ModelInputs):
                Inputs to the model
            model_outputs (:obj:`Any`):
                Output of the model

        Returns:

        """
        raise NotImplementedError

    def preprocess_text(
        self,
        text: Union[str, List[str], List[List[str]], List[Word], List[List[Word]]],
        is_split_into_words,
    ) -> List[List[Word]]:
        """
        Preprocess the input text, performing tokenization, pos tagging and lemmatization if specified
        by the user.

        Args:
            text (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`, :obj:`List[Word]`, :obj:`List[List[Word]]`):
                Text in input. It can be a single string, a batch of string, text that is already tokenized
                or text already preprocessed.
            is_split_into_words (:obj:`bool`):
                :obj:`True` means that the input is already tokenized into single words.
        Returns:
            :obj:`List[List[Word]]`: the text in input preprocessed into a batch of :obj:`Word`.
        """
        # check if input is raw text
        is_str = isinstance(text, str)
        # check if it is a single pre-tokenized text
        is_str_split = (
            is_split_into_words and isinstance(text, list) and text and isinstance(text[0], str)
        )
        # check if it is a batch of raw text
        is_str_batch = (
            not is_split_into_words and isinstance(text, list) and text and isinstance(text[0], str)
        ) or (
            is_split_into_words
            and isinstance(text, list)
            and text
            and isinstance(text[0], list)
            and isinstance(text[0][0], str)
        )
        if is_str or is_str_split or is_str_batch:
            if not self.tokenizer:
                raise TypeError("`self.tokenizer` is not instantiated.")
            # to avoid increase memory footprint, the tokenizer is
            # loaded only when needed
            try:
                text = self.tokenizer(text, is_split_into_words=is_split_into_words)
            except TypeError:
                # lazy load the tokenizer
                self.tokenizer = self.tokenizer(**self.tokenizer_kwargs)
                text = self.tokenizer(text, is_split_into_words=is_split_into_words)
        # if is a single sample, normalize to batch of length 1
        if isinstance(text, list) and isinstance(text[0], Word):
            text = [text]
        return text

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_dir: Union[str, Path],
        device: str = "cpu",
        strict: bool = True,
        onnx: bool = False,
        predictor_name: str = None,
        **kwargs,
    ):
        """
        Load a predictor from a pre-trained model available as model card.

        Args:
            model_name_or_dir (:obj:`str`)
                Name of the model to load.
            device(:obj:`str`, `optional`, defaults to :obj:`cpu`):
                Device where to move the loaded model (:obj:`cpu` or :obj:`cuda`).
            strict (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether to strictly enforce that the keys in the model weights
                match the keys returned by its module.
            onnx (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to load the ONNX version of the model, if available.
            predictor_name(:obj:`str`, `optional`):
                Classname of the predictor for the model. If none, it will use the default
                predictor for the model.
        Returns:
            :obj:`Predictor`: The predictor specified by :obj:`model_name`.
        """

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)

        if is_remote_url(model_name_or_dir):
            # if model_name_or_dir is an URL
            # download it and try to load
            model_archive = model_name_or_dir
        elif Path(model_name_or_dir).is_dir() or Path(model_name_or_dir).is_file():
            # if model_name_or_dir is a local directory or
            # an archive file try to load it
            model_archive = model_name_or_dir
        else:
            # probably model_name_or_dir is a sapienzanlp model id
            # guess the url and try to download
            model_name_or_dir_ = model_name_or_dir
            if onnx:
                # usually onnx version of the models are
                # stored with `-onnx` append to the end
                model_name_or_dir_ += "-onnx"
            model_archive = sapienzanlp_model_urls(model_name_or_dir_)

        try:
            # Load from URL or cache if already cached
            cached_model = from_cache(
                model_archive,
                cache_dir=cache_dir,
                force_download=force_download,
            )
        except EnvironmentError as err:
            logger.error(err)
            if onnx:
                raise EnvironmentError(
                    f"Can't load onnx version of '{model_name_or_dir}', check if:\n"
                    f"- '{model_name_or_dir}' has a ONNX version\n"
                    f"- '{model_name_or_dir}' is a correct model identifier (actual url TODO)\n"
                    f"- or '{model_name_or_dir}' is the correct path to a model archive or directory\n"
                )
            else:
                raise EnvironmentError(
                    f"Can't load '{model_name_or_dir}', check if:\n"
                    f"- '{model_name_or_dir}' is a correct model identifier (actual url TODO)\n"
                    f"- or '{model_name_or_dir}' is the correct path to a model archive or directory\n"
                )

        if not (Path(cached_model) / WEIGHTS_NAME).exists():
            raise OSError(
                f"there is no {WEIGHTS_NAME} file in the model directory. Cannot load model."
            )
        model = Model.load(cached_model, device, strict)
        model.eval()

        if predictor_name is None:
            predictor_name = model.default_predictor
        logger.debug(f"Loading predictor: {predictor_name}")
        predictor = Predictor.from_name(predictor_name)
        return predictor(model, **kwargs)
