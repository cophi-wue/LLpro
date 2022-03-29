from __future__ import annotations

from abc import abstractmethod
from typing import TextIO, Iterable, Sequence, List

import more_itertools
from tqdm import tqdm


class Token:
    fields = "doc id word sentence lemma pos morph".split()
    id: int
    doc: str
    word: str
    sentence: int
    lemma: str
    pos: str
    morph: str
    head: int
    deprel: str

    def __init__(self):
        self.fields = {}

    def get_field(self, key, module_name=None):
        if module_name is not None:
            return self.fields[(key, module_name)]

        candidates = [(field_key, field_module_name) for field_key, field_module_name in self.fields.keys() if
                      field_key == key]
        if len(candidates) == 0:
            raise TypeError(f'Field {key} not set')
        if len(candidates) > 1:
            raise TypeError(f'Field {key} set by multiple modules; call get_field(field, module_name)')
        return self.fields[candidates[0]]

    def set_field(self, field, module_name, value):
        self.fields[(field, module_name)] = value

    def __setattr__(self, key, value):
        if key in Token.fields:
            raise TypeError('Fields need to be set with set_field(field, module_name)')
        else:
            object.__setattr__(self, key, value)

    def __getattribute__(self, key):
        if key in Token.fields:
            return self.get_field(key)
        else:
            return object.__getattribute__(self, key)

    def __str__(self):
        return self.fields.__str__()

    def to_output_line(self, modules=None, fields=None):
        if fields is None:
            fields = ['doc', 'sentence', 'id', 'word', 'lemma', 'pos', 'morph', 'head', 'deprel']

        if modules is None:
            modules = {}

        field_strings = [str(self.get_field(field, module_name=modules.get(field, None))) for field in fields]
        return '\t'.join(field_strings)

    @staticmethod
    def get_sentences(tokens: Iterable[Token]) -> Iterable[Iterable[Token]]:
        return more_itertools.split_when(tokens, lambda a, b: a.sentence != b.sentence)

    @staticmethod
    def get_documents(tokens: Iterable[Token]) -> Iterable[Iterable[Token]]:
        return more_itertools.split_when(tokens, lambda a, b: a.doc != b.doc)


class Tokenizer:
    def tokenize(self, file: TextIO, filename: str) -> Iterable[Token]:
        raise NotImplementedError


class Module:

    def run(self, tokens: Iterable[Token], pbar: tqdm = None, pbar_opts=None, **kwargs):
        tokens = list(tokens)

        if pbar is None:
            pbar_opts = pbar_opts if pbar_opts is not None else {}
            pbar = tqdm(total=len(tokens), unit='tok', postfix=type(self).__name__, **pbar_opts)

        def my_update_fn(x: int):
            pbar.update(x)

        self.process(tokens, my_update_fn, **kwargs)
        pbar.close()

    @abstractmethod
    def process(self, tokens: List[Token], update_fn, **kwargs):
        raise NotImplementedError
