from pathlib import Path
from typing import Union, List, Set, Dict

from overrides import overrides

from sapienzanlp.common.from_config import FromConfig
from sapienzanlp.common.utils import dump_json


class Labels(FromConfig):
    """
    Class that contains the labels for a model.

    Args:
        _labels_to_index (:obj:`Dict[str, Dict[str, int]]`):
            A dictionary from :obj:`str` to :obj:`int`.
        _index_to_labels (:obj:`Dict[str, Dict[int, str]]`):
            A dictionary from :obj:`int` to :obj:`str`.

    """

    def __init__(
        self,
        _labels_to_index: Dict[str, Dict[str, int]] = None,
        _index_to_labels: Dict[str, Dict[int, str]] = None,
        **kwargs,
    ):
        self._labels_to_index = _labels_to_index or {}
        self._index_to_labels = _index_to_labels or {}
        # if _labels_to_index is not empty and _index_to_labels is not provided
        # to the constructor, build the inverted label dictionary
        if not _index_to_labels and _labels_to_index:
            for namespace in self._labels_to_index:
                self._index_to_labels[namespace] = {
                    v: k for k, v in self._labels_to_index[namespace].items()
                }

    def get_index_from_label(self, label: str, namespace: str = "labels") -> int:
        """
        Returns the index of a literal label.

        Args:
            label (:obj:`str`):
                The string representation of the label.
            namespace (:obj:`str`, optional, defaults to ``labels``):
                The namespace where the label belongs, e.g. ``roles`` for a SRL task.

        Returns:
            :obj:`int`: The index of the label.
        """
        if namespace not in self._labels_to_index:
            raise ValueError(f"Provided namespace {namespace} is not in the label dictionary.")

        if label not in self._labels_to_index[namespace]:
            raise ValueError(f"Provided label {label} is not in the label dictionary.")

        return self._labels_to_index[namespace][label]

    def get_label_from_index(self, index: int, namespace: str = "labels") -> str:
        """
        Returns the string representation of the label index.

        Args:
            index (:obj:`int`):
                The index of the label.
            namespace (:obj:`str`, optional, defaults to ``labels``):
                The namespace where the label belongs, e.g. ``roles`` for a SRL task.

        Returns:
            :obj:`str`: The string representation of the label.
        """
        if namespace not in self._index_to_labels:
            raise ValueError(f"Provided namespace {namespace} is not in the label dictionary.")

        if index not in self._index_to_labels[namespace]:
            raise ValueError(f"Provided label {index} is not in the label dictionary.")

        return self._index_to_labels[namespace][index]

    def add_labels(
        self, labels: Union[str, List[str], Set[str], Dict[str, int]], namespace: str
    ) -> List[int]:
        """
        Adds the labels in input in the label dictionary.

        Args:
            labels (:obj:`str`, :obj:`List[str]`, :obj:`Set[str]`):
                The labels (single label, list of labels or set of labels) to add to the dictionary.
            namespace (:obj:`str`, optional, defaults to ``labels``):
                Namespace where the labels belongs.

        Returns:
            :obj:`List[int]`: The index of the labels just inserted.

        """
        if isinstance(labels, dict):
            self._labels_to_index[namespace] = labels
            self._index_to_labels[namespace] = {
                v: k for k, v in self._labels_to_index[namespace].items()
            }
        # normalize input
        if isinstance(labels, (str, list)):
            labels = set(labels)
        # if new namespace, add to the dictionaries
        if namespace not in self._labels_to_index:
            self._labels_to_index[namespace] = {}
            self._index_to_labels[namespace] = {}
        # returns the new indices
        return [self._add_label(label, namespace) for label in labels]

    def _add_label(self, label: str, namespace: str = "labels") -> int:
        """
        Adds the label in input in the label dictionary.

        Args:
            label (:obj:`str`):
                The label to add to the dictionary.
            namespace (:obj:`str`, optional, defaults to ``labels``):
                Namespace where the label belongs.

        Returns:
            :obj:`List[int]`: The index of the label just inserted.

        """
        if label not in self._labels_to_index[namespace]:
            index = len(self._labels_to_index[namespace])
            self._labels_to_index[namespace][label] = index
            self._index_to_labels[namespace][index] = label
            return index
        else:
            return self._labels_to_index[namespace][label]

    def get_labels(self, namespace: str = "labels") -> Dict[str, int]:
        """
        Returns all the labels that belongs to the input namespace.

        Args:
            namespace (:obj:`str`, optional, defaults to ``labels``):
                Labels namespace to retrieve.

        Returns:
            :obj:`Dict[str, int]`: The label dictionary, from ``str`` to ``int``.
        """
        if namespace not in self._labels_to_index:
            raise ValueError(f"Provided namespace {namespace} is not in the label dictionary.")
        return self._labels_to_index[namespace]

    def get_label_size(self, namespace: str = "labels") -> int:
        """
        Returns the number of the labels in the namespace dictionary.

        Args:
            namespace (:obj:`str`, optional, defaults to ``labels``):
                Labels namespace to retrieve.

        Returns:
            :obj:`int`: Number of labels.

        """
        if namespace not in self._labels_to_index:
            raise ValueError(f"Provided namespace {namespace} is not in the label dictionary.")
        return len(self._labels_to_index[namespace])

    @overrides
    def to_config(self, config_path: Union[str, Path, dict], name: str = None, **kwargs):
        name = name or type(self).__name__
        config = {
            "name": name,
            "class": f"{type(self).__module__}.{type(self).__name__}",
            "_labels_to_index": self._labels_to_index,
        }
        dump_json(config, config_path)
