from __future__ import annotations

import itertools
import multiprocessing
import os
import time
from abc import abstractmethod, ABC
from typing import TextIO, Iterable, Sequence, List

import more_itertools
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


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

    def __init__(self, fields=None):
        self.fields = {}
        if fields is not None:
            for (key, module_name), value in fields.items():
                self.set_field(key, module_name, value)

    def get_field(self, key, module_name=None, default=None):
        if module_name is not None:
            return self.fields[(key, module_name)]

        candidates = [(field_key, field_module_name) for field_key, field_module_name in self.fields.keys() if
                      field_key == key]
        if len(candidates) > 1:
            raise TypeError(f'Field {key} set by multiple modules; call get_field(field, module_name)')
        if len(candidates) == 0:
            if default is None:
                raise TypeError(f'Field {key} not set')
            else:
                return '_'
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

        field_strings = [str(self.get_field(field, module_name=modules.get(field, None), default='_')) for field in
                         fields]
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

    def name(self):
        return type(self).__name__

    def run(self, tokens: Iterable[Token], pbar: tqdm = None, pbar_opts=None, **kwargs):
        tokens = list(tokens)
        if pbar is None:
            pbar_opts = pbar_opts if pbar_opts is not None else {}
            pbar = tqdm(total=len(tokens), unit='tok', postfix=self.name(), **pbar_opts)

        def my_update_fn(x: int):
            pbar.update(x)

        with logging_redirect_tqdm():
            self.process(tokens, my_update_fn, **kwargs)
            pbar.update(len(tokens) - pbar.n)
            pbar.close()

    @abstractmethod
    def process(self, tokens: List[Token], update_fn, **kwargs):
        raise NotImplementedError


_WORKER_MODULE: Module = None


class ParallelizedModule(Module):

    def __init__(self, module, num_processes=4, chunking='sentences', chunks_per_process=1):
        if type(module) is type:
            self._name = module.__name__ + 'x' + str(num_processes)
        else:
            self._name = type(self).__name__ + 'x' + str(num_processes)

        self.chunking = chunking
        self.chunks_per_process = chunks_per_process
        self.pool = multiprocessing.Pool(processes=num_processes, initializer=ParallelizedModule._init_worker,
                                         initargs=(module,))

    def name(self):
        return self._name

    def process(self, tokens: List[Token], update_fn, **kwargs):
        # chunked = Token.get_sentences(tokens)
        # self.pool.map_async()
        m = multiprocessing.Manager()
        q = m.Queue()
        chunks = list(Token.get_sentences(tokens))
        process_chunks = [itertools.chain.from_iterable(it) for it in more_itertools.chunked(chunks, n=self.chunks_per_process)]
        for i, chunk in enumerate(process_chunks):
            self.pool.apply_async(ParallelizedModule._process_worker, (list(chunk), i, q))
        processed_chunks = [None] * len(process_chunks)
        while True:
            kind, i, value = q.get()
            if kind == 'update':
                update_fn(value)
            elif kind == 'result':
                processed_chunks[i] = value
                if not any(x is None for x in processed_chunks):
                    break

        for tok, modified_tok in zip(tokens, itertools.chain.from_iterable(processed_chunks)):
            tok.fields = modified_tok.fields

    @staticmethod
    def _init_worker(module_constructor):
        global _WORKER_MODULE
        _WORKER_MODULE = module_constructor()

    @staticmethod
    def _process_worker(tokens, i, out_queue):
        global _WORKER_MODULE

        def send_update(x):
            out_queue.put(('update', i, x))

        _WORKER_MODULE.process(tokens, send_update)
        out_queue.put(('result', i, tokens))
        return
