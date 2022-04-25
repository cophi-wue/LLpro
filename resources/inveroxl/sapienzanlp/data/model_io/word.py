from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Word:
    """
    A word representation that includes text, index in the sentence, POS tag, lemma,
    dependency relation, and similar information.

    # Parameters
    text : `str`, optional
        The text representation.
    index : `int`, optional
        The word offset in the sentence.
    lemma : `str`, optional
        The lemma of this word.
    pos : `str`, optional
        The coarse-grained part of speech of this word.
    dep : `str`, optional
        The dependency relation for this word.

    input_id : `int`, optional
        Integer representation of the word, used to pass it to a model.
    token_type_id : `int`, optional
        Token type id used by some transformers.
    attention_mask: `int`, optional
        Attention mask used by transformers, indicates to the model which tokens should
        be attended to, and which should not.
    """

    text: str
    index: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    # preprocessing fields
    lemma: Optional[str] = None
    pos: Optional[str] = None
    dep: Optional[str] = None

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()


# Word Sense Disambiguation stuff


@dataclass
class WsdWord(Word):
    """
    Word Sense Disambiguation word. It includes all the fields from `Word` plus its synset from various
    inventories.

    # Parameters
    babelnet_synset : `str`, optional
        The label as BabelNet synset.
    wordnet_synset : `str`, optional
        The label as WordNet synset.
    nltk_synset : `str`, optional
        The label as WordNet synset in the NLTK format.

    """

    babelnet_synset: str = "O"
    wordnet_synset: str = "O"
    nltk_synset: str = "O"

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()


# Semantic Role Labeling stuff


@dataclass
class Predicate(Word):
    """
    Semantic Role Labeling predicate word. It includes all the fields from `Word` plus
    predicate-related fields from Semantic Role Labeling.

    # Parameters
    sense : `str`, optional
        The label of the predicate word.
    arguments : `list`, optional
        The list of the arguments of the predicate word.
    """

    sense: Optional[str] = None
    arguments: Optional[List[Argument]] = field(default_factory=list)

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()


class Argument:
    """
    Semantic Role Labeling argument span.

    # Parameters
    role : `str`
        The label of the argument span.
    predicate : `Predicate`
        The predicate that has this argument span.
    words : `list`
        The list of words that take part in the argument span.
    start_index : `int`
        The start index of the argument span in the sentence.
    end_index : `int`
        The end index of the argument span in the sentence.
    """

    def __init__(
        self, role: str, predicate: Predicate, words: List[Word], start_index: int, end_index: int
    ):
        """

        Args:
            role:
            predicate:
            words:
            start_index:
            end_index:
        """
        self.role: str = role
        self.predicate: Predicate = predicate
        self.words: List[Word] = words
        self.start_index: int = start_index
        self.end_index: int = end_index

    @property
    def span(self):
        return self.start_index, self.end_index
