"""
Collection of dependency parser integrations.
"""
from enum import Enum

from .spacy_herma import HermaParser
from .spacy_parzu import ParZuParser


class Parser(Enum):
    SPACY = "spacy"
    HERMA = "herma"
    PARZU = "parzu"
