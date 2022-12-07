import logging
import pickle
import regex as re
from functools import partial
import sys
from pathlib import Path

import torch
from spacy import Language
from spacy.morphology import Morphology
from spacy.tokens import Doc, Token, MorphAnalysis
from typing import Callable

from ..common import Module


@Language.factory("tagger_rnntagger", assigns=['token._.rnntagger_tag', 'token.morph'], default_config={
    'rnntagger_home': 'resources/RNNTagger', 'use_cuda': True, 'pbar_opts': None
})
def tagger_rnntagger(nlp, name, rnntagger_home):
    if not Token.has_extension('rnntagger_tag'):
        Token.set_extension('rnntagger_tag', default='')
    return RNNTagger(name=name, rnntagger_home=rnntagger_home)


class RNNTagger(Module):

    def __init__(self, name, rnntagger_home='resources/RNNTagger', use_cuda=True, pbar_opts=None):
        super().__init__(name, pbar_opts=pbar_opts)
        self.rnntagger_home = Path(rnntagger_home)
        sys.path.insert(0, str(self.rnntagger_home))
        sys.path.insert(0, str(self.rnntagger_home / "PyNMT"))
        from PyRNN.Data import Data
        import PyRNN.RNNTagger
        import PyRNN.CRFTagger

        with open(str(self.rnntagger_home / "lib/PyRNN/german.hyper"), "rb") as file:
            hyper_params = pickle.load(file)
        self.vector_mappings = Data(str(self.rnntagger_home / "lib/PyRNN/german.io"))
        self.model = PyRNN.CRFTagger.CRFTagger(*hyper_params) if len(hyper_params) == 10 \
            else PyRNN.RNNTagger.RNNTagger(*hyper_params)
        self.model.load_state_dict(torch.load(str(self.rnntagger_home / "lib/PyRNN/german.rnn")))
        if torch.cuda.is_available() and use_cuda:
            self.model = self.model.cuda()
        self.model.eval()
        logging.info(f"RNNTagger using device {next(self.model.parameters()).device}")

        def annotate_sentence(words):
            data = self.vector_mappings
            # vgl. RNNTagger/PyRNN/rnn-annotate.py
            fwd_charIDs, bwd_charIDs = data.words2charIDvec(words)
            fwd_charIDs = self.model.long_tensor(fwd_charIDs)
            bwd_charIDs = self.model.long_tensor(bwd_charIDs)

            word_embs = None if data.word_emb_size <= 0 else self.model.float_tensor(data.words2vecs(words))

            with torch.no_grad():
                if type(self.model) is PyRNN.RNNTagger.RNNTagger:
                    tagscores = self.model(fwd_charIDs, bwd_charIDs, word_embs)
                    softmax_probs = torch.nn.functional.softmax(tagscores,
                                                                dim=-1)  # ae: added softmax transform to get meaningful probabilities
                    best_prob, tagIDs = softmax_probs.max(dim=-1)
                    tags = data.IDs2tags(tagIDs)
                    return [{'tag': t, 'prob': p.item()} for t, p in zip(tags, best_prob.cpu())]
                elif type(self.model) is PyRNN.CRFTagger.CRFTagger:
                    tagIDs = self.model(fwd_charIDs, bwd_charIDs, word_embs)
                    return [{'tag': t} for t in data.IDs2tags(tagIDs)]
                else:
                    raise RuntimeError("Error in function annotate_sentence")

        self.annotate_sentence = annotate_sentence

    def process(self, doc: Doc, progress_fn: Callable[[int], None]) -> Doc:
        for sent in doc.sents:
            sent = list(sent)
            for token, out in zip(sent, self.annotate_sentence([t.text for t in sent])):
                if 'prob' in out.keys():
                    tigertag, prob = out['tag'], out['prob']
                else:
                    tigertag = out['tag']
                    prob = None

                token._.rnntagger_tag = tigertag

                morph = token.morph.to_dict()
                morph.update(from_tigertag(tigertag))
                token.set_morph(MorphAnalysis(token.vocab, morph))
                progress_fn(1)

        return doc


def feature(featname, arg):
    """Use this dict to change the way morph values
    are represented.
    Map a value to `None` if you want a feature to
    be removed."""
    value_map = {'Sg': 'Sing', 'Pl': 'Plur'}
    return (featname, value_map.get(arg, arg))


degree = partial(feature, "Degree")
case = partial(feature, "Case")
number = partial(feature, "Number")
gender = partial(feature, "Gender")
definite = partial(feature, "Definite")
person = partial(feature, "Person")
tense = partial(feature, "Tense")
mood = partial(feature, "Mood")


def from_tigertag(tigertag):
    """Extract morphological information from a TIGER tag."""
    stts, *parts = tigertag.split(".")
    if stts == '$':
        stts = '$.'

    # Maps an POS tag to the list of functions we need in
    # order to interpret the morphological information.
    def fields_for_tag(tag):
        if tag == "ADJA":
            return [degree, case, number, gender]
        elif tag == "ADJD":
            return [degree]
        elif tag in {"VVFIN", "VAFIN", "VMFIN"}:
            return [person, number, tense, mood]
        elif tag in {"VVIMP", "VAIMP"}:
            return [person, number, mood]
        elif tag in {
            "APPRART",
            "ART",
            "NN",
            "NE",
            "PPOSAT",
            "PPOSS",
            "PDAT",
            "PDS",
            "PIAT",
            "PIS",
            "PRELS",
            "PRELAT",
            "PWAT",
            "PWS",
        }:
            return [case, number, gender]
        elif tag == "PPER":
            return [person, case, number, gender]
        elif tag == "PRF":
            return [person, case, number]
        else:
            return []

    active_fields = fields_for_tag(stts)
    feats = dict(f(p) for f, p in zip(active_fields, parts))

    return feats
