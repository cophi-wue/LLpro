import logging

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """
    The exception raised when the configuration file is wrong.
    """

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def __str__(self):
        return self.message
