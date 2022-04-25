import importlib
from pathlib import Path
from typing import Union, Type, TypeVar, Dict, Any

from sapienzanlp.common.logging import get_logger
from sapienzanlp.common.utils import load_json, dump_json

logger = get_logger()

T = TypeVar("T")


class FromConfig:
    """
    This class gives the method :obj:`from_config`.
    It allows to load recursively the objects defined in a configuration file.
    """

    @classmethod
    def from_config(cls: Type[T], config: Union[str, Path, dict], **kwargs) -> T:
        """
        Load class configuration from file.

        Args:
            config (:obj:`str`, :obj:`Path`, :obj:`dict`):
                Path (or dictionary) of the config to load.

        Returns:
            :obj:`T`: The class.
        """
        if isinstance(config, (str, Path)):
            config = load_json(config)
        subclass = cls.resolve_class_name(config.pop("class"))
        args = subclass.get_args(config, **kwargs)
        instantiated = subclass(**args)
        return instantiated

    @classmethod
    def get_args(cls: Type[T], config: dict, **kwargs) -> Dict[str, Any]:
        """
        Parse the arguments of the class. If an argument is itself a class, it loads it recursively from
        the config file.

        Args:
            config (:obj:`dict`):
                Config to load.

        Returns:
            :obj:`T`: The class.
        """
        class_args = {}
        for param, value in config.items():
            if isinstance(value, dict) and "class" in value:
                # the parameter is a class, instantiate it
                class_args[param] = cls.from_config(value)
            else:
                class_args[param] = value
        # add additional parameters passed in kwargs
        for param, value in kwargs.items():
            class_args[param] = value
        return class_args

    @classmethod
    def from_name(cls: Type[T], name: str) -> T:
        """
        Returns a callable function that constructs an argument of the class.

        Args:
            name (:obj:str):
                Complete class name. E.g. my.package.ClassName.

        Returns:
            :obj:`T`: The class.

        """
        logger.debug(f"instantiating registered subclass {name} of {cls}")
        subclass = cls.resolve_class_name(name)
        return subclass

    @classmethod
    def to_config(cls: Type[T], config_path: Union[str, Path, dict], name: str = None, **kwargs):
        """
        Store class configuration in a file.

        Args:
            config_path (:obj:`str`, :obj:`Path`, :obj:`dict`):
                Path (or dictionary) of the config to store as a file.
            name (:obj:`str`, optional):
                The name of the class to store.

        Returns:

        """
        name = name or type(cls).__name__
        config = {"name": name, "class": f"{type(cls).__module__}.{type(cls).__name__}"}
        dump_json(config, config_path)

    @classmethod
    def resolve_class_name(cls: Type[T], name: str) -> T:
        """
        Resolve class by name.

        Args:
            name (:obj:str):
                Complete class name. E.g. my.package.ClassName.

        Returns:
            :obj:`T`: The class.

        """
        # split name in parts
        parts = name.split(".")
        # combine submodule
        submodule = ".".join(parts[:-1])
        # get class name
        class_name = parts[-1]
        # import submodule
        module = importlib.import_module(submodule)
        # retrieve the class from the submodule
        subclass = getattr(module, class_name)
        return subclass
