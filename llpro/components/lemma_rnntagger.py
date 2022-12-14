import logging
import pickle
import regex as re
import sys
from pathlib import Path

from spacy import Language
from spacy.tokens import Token, Doc
from typing import Callable

import torch

from ..common import Module


@Language.factory("lemma_rnntagger", requires=['token._.rnntagger_tag'], assigns=['token.lemma'], default_config={
    'rnntagger_home': 'resources/RNNTagger', 'use_cuda': True, 'device_on_run': False, 'pbar_opts': None
})
def lemma_rnntagger(nlp, name, rnntagger_home, use_cuda, device_on_run, pbar_opts):
    if not Token.has_extension('rnntagger_tag'):
        Token.set_extension('rnntagger_tag', default='')
    return RNNLemmatizer(name=name, rnntagger_home=rnntagger_home, use_cuda=use_cuda, device_on_run=device_on_run, pbar_opts=pbar_opts)


class RNNLemmatizer(Module):

    def __init__(self, name, rnntagger_home='resources/RNNTagger', use_cuda=True, device_on_run=False, pbar_opts=None):
        super().__init__(name, pbar_opts=pbar_opts)
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else "cpu")
        self.device_on_run = device_on_run
        self.rnntagger_home = Path(rnntagger_home)
        sys.path.insert(0, str(self.rnntagger_home))
        sys.path.insert(0, str(self.rnntagger_home / "PyNMT"))
        from PyNMT.Data import Data, rstrip_zeros
        from PyNMT.NMT import NMTDecoder

        beam_size = 0
        batch_size = 32
        self.vector_mappings = Data(str(self.rnntagger_home / "lib/PyNMT/german.io"), batch_size)

        with open(str(self.rnntagger_home / "lib/PyNMT/german.hyper"), "rb") as file:
            hyper_params = pickle.load(file)
        self.model = NMTDecoder(*hyper_params)
        self.model.load_state_dict(torch.load(str(self.rnntagger_home / "lib/PyNMT/german.nmt")))
        if torch.cuda.is_available() and use_cuda:
            self.model = self.model.cuda()
        self.model.eval()
        logging.info(f"RNNLemmatizer using device {next(self.model.parameters()).device}")

        def process_batch(batch):
            # see RNNTagger/PyNMT/nmt-translate.py
            src_words, sent_idx, (src_wordIDs, src_len) = batch
            with torch.no_grad():
                tgt_wordIDs, tgt_logprobs = self.model.translate(src_wordIDs, src_len, beam_size)
            # undo the sorting of sentences by length
            tgt_wordIDs = [tgt_wordIDs[i] for i in sent_idx]

            for swords, twordIDs, logprob in zip(src_words, tgt_wordIDs, tgt_logprobs):
                twords = self.vector_mappings.target_words(rstrip_zeros(twordIDs))
                yield ''.join(twords), torch.exp(logprob).cpu().item()

        def format_batch(tokens):
            # see RNNTagger/scripts/reformat.pl
            batch = []
            for tok in tokens:
                word = tok.text
                tag = tok._.rnntagger_tag
                word = re.sub(r'   ', ' <> ', re.sub(r'(.)', r'\g<1> ', word))
                tag = re.sub(r'(.)', r'\g<1> ', tag)

                formatted = word + ' ## ' + tag + '\n'
                batch.append(formatted.split())
            return self.vector_mappings.build_test_batch(batch)

        def myprocessor(tokens_batch):
            assert len(tokens_batch) <= self.vector_mappings.batch_size
            for out, prob in process_batch(format_batch(list(tokens_batch))):
                yield out, prob

        self.processor = myprocessor

    def before_run(self):
        if self.device_on_run:
            self.model.to(self.device)
            logging.info(f"{self.name} using device {next(self.model.parameters()).device}")

    def after_run(self):
        if self.device_on_run:
            self.model.to('cpu')
            torch.cuda.empty_cache()

    def process(self, doc: Doc, progress_fn: Callable[[int], None]) -> Doc:
        cached = {}
        it = iter(doc)
        done = False

        while not done:
            current_batch = []
            current_batch_is_cached = []
            try:
                while sum(1 - x for x in current_batch_is_cached) < self.vector_mappings.batch_size:
                    tok = next(it)
                    cache_key = (tok.text, tok._.rnntagger_tag)
                    current_batch.append(tok)
                    if cache_key in cached.keys():
                        current_batch_is_cached.append(1)
                    else:
                        current_batch_is_cached.append(0)
            except StopIteration:
                done = True
                pass

            processed = iter(self.processor(
                [tok for tok, is_cached in zip(current_batch, current_batch_is_cached) if not is_cached]))
            for tok, is_cached in zip(current_batch, current_batch_is_cached):
                cache_key = (tok.text, tok._.rnntagger_tag)
                if is_cached:
                    lemma, prob = cached[cache_key]
                    tok.lemma_ = lemma
                else:
                    lemma, prob = next(processed)
                    cached[cache_key] = (lemma, prob)
                    tok.lemma_ = lemma
                progress_fn(1)
        return doc
