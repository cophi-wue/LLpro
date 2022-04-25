import json
import logging
import os
from pathlib import Path
from typing import Dict, Union, Optional

import psutil
import torch
import torch.nn as nn

from sapienzanlp.common.exception import ConfigurationError
from sapienzanlp.common.from_config import FromConfig
from sapienzanlp.common.logging import get_logger
from sapienzanlp.common.utils import WEIGHTS_NAME, LABELS_NAME, CONFIG_NAME, load_json
from sapienzanlp.data.labels import Labels

logger = get_logger(level=logging.DEBUG)


class Model(nn.Module, FromConfig):
    def __init__(
        self,
        labels: Union[Labels, Dict],
        device: Union[str, torch.device] = None,
        default_predictor: Optional[str] = "sapienzanlp.predictors.Predictor",
    ):
        super(Model, self).__init__()
        self.labels = labels
        self.device = device
        self.default_predictor = default_predictor

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @classmethod
    def load(
            cls,
            model_dir: Union[str, Path],
            device: str = "cpu",
            strict: bool = True,
    ):
        num_threads = os.getenv("TORCH_NUM_THREADS", psutil.cpu_count(logical=False))
        torch.set_num_threads(num_threads)
        logger.info(f"Model is running on {num_threads} threads")
        # get model stuff
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        weights_path = model_dir / WEIGHTS_NAME
        config_path = model_dir / CONFIG_NAME
        # load model labels
        labels = {}
        logger.info(f"Loading labels")
        for inventory in ["ca", "cz", "de", "en", "es", "va", "zh"]:
            labels[inventory] = Labels.from_config(model_dir / "labels_{}.json".format(inventory))
        # parse config file
        params = load_json(config_path)
        # load model from config
        if "model" not in params:
            raise ConfigurationError(f"Configuration file doesn't contains `model` key.")
        logger.debug("Loading model")
        model = Model.from_config(params.get("model"), labels=labels)
        # move model to device
        model.device = device
        # load model weights
        model_state = torch.load(weights_path, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=strict)
        if unexpected_keys or missing_keys:
            logger.debug(
                f"Error loading state dict for {model.__class__.__name__}\n\t"
                f"Missing keys: {missing_keys}\n\t"
                f"Unexpected keys: {unexpected_keys}"
            )
        # quantization
        # model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        model.to(device)
        return model
