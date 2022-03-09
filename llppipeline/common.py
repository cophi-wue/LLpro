from __future__ import annotations
from typing import TextIO, Iterable, Sequence

import more_itertools

class Token:
    fields = "doc id word sentence lemma pos morph".split()

    def __init__(self, **kwargs):
        for key in Token.fields:
            # covers unspecified fields
            val = kwargs.get(key, "_")
            # covers fields specified as None
            self.__dict__[key] = val if val else "_"

    def __str__(self):
        return "\t".join(str(self.__dict__.get(key) or '_') for key in Token.fields)

    @staticmethod
    def get_sentences(tokens: Iterable[Token]) -> Iterable[Iterable[Token]]:
        return more_itertools.split_when(tokens, lambda a, b: a.sentence != b.sentence)


class Tokenizer:
    def tokenize(self, file: TextIO, filename: str) -> Iterable[Token]:
        raise NotImplementedError


class Module:
    def process(self, tokens: Sequence[Token], **kwargs):
        raise NotImplementedError
