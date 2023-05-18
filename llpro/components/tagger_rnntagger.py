import logging
import pickle
from functools import partial
import sys
from pathlib import Path

import torch
from spacy import Language, Vocab
from spacy.tokens import Doc, Token, MorphAnalysis
from typing import Callable

from ..common import Module
from ..stts2upos import conv_table
from .. import LLPRO_RESOURCES_ROOT

logger = logging.getLogger(__name__)


@Language.factory("tagger_rnntagger", assigns=['token._.rnntagger_tag', 'token.morph'], default_config={
    'rnntagger_home': LLPRO_RESOURCES_ROOT + '/RNNTagger', 'use_cuda': True, 'device_on_run': True, 'pbar_opts': None
})
def tagger_rnntagger(nlp, name, rnntagger_home, use_cuda, device_on_run, pbar_opts):
    if not Token.has_extension('rnntagger_tag'):
        Token.set_extension('rnntagger_tag', default='')
    return RNNTagger(name=name, rnntagger_home=rnntagger_home, use_cuda=use_cuda, device_on_run=device_on_run, pbar_opts=pbar_opts)


class RNNTagger(Module):

    def __init__(self, name, rnntagger_home=LLPRO_RESOURCES_ROOT + '/RNNTagger', use_cuda=True, device_on_run=True, pbar_opts=None):
        super().__init__(name, pbar_opts=pbar_opts)
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else "cpu")
        self.device_on_run = device_on_run
        self.rnntagger_home = Path(rnntagger_home)
        sys.path.insert(0, str(self.rnntagger_home))
        sys.path.insert(0, str(self.rnntagger_home / "PyRNN"))
        from PyRNN.Data import Data
        import PyRNN.RNNTagger
        import PyRNN.CRFTagger

        with open(str(self.rnntagger_home / "lib/PyRNN/german.hyper"), "rb") as file:
            hyper_params = pickle.load(file)
        self.vector_mappings = Data(str(self.rnntagger_home / "lib/PyRNN/german.io"))
        self.model = PyRNN.CRFTagger.CRFTagger(*hyper_params) if len(hyper_params) == 10 \
            else PyRNN.RNNTagger.RNNTagger(*hyper_params)
        self.model.load_state_dict(torch.load(str(self.rnntagger_home / "lib/PyRNN/german.rnn"), map_location=self.device))
        self.model.eval()

        if not self.device_on_run:
            self.model.to(self.device)
            logger.info(f"{self.name} using device {next(self.model.parameters()).device}")

        def annotate_sentence(words):
            data = self.vector_mappings
            # cf. RNNTagger/PyRNN/rnn-annotate.py
            fwd_charIDs, bwd_charIDs = data.words2charIDvec(words)
            fwd_charIDs = torch.LongTensor(fwd_charIDs).to(self.device)
            bwd_charIDs = torch.LongTensor(bwd_charIDs).to(self.device)

            with torch.no_grad():
                if type(self.model) is PyRNN.RNNTagger.RNNTagger:
                    tagscores = self.model(fwd_charIDs, bwd_charIDs)
                    softmax_probs = torch.nn.functional.softmax(tagscores,
                                                                dim=-1)  # ae: added softmax transform to get meaningful probabilities
                    best_prob, tagIDs = softmax_probs.max(dim=-1)
                    tags = data.IDs2tags(tagIDs)
                    return [{'tag': t, 'prob': p.item()} for t, p in zip(tags, best_prob.cpu())]
                elif type(self.model) is PyRNN.CRFTagger.CRFTagger:
                    tagIDs = self.model(fwd_charIDs, bwd_charIDs)
                    return [{'tag': t} for t in data.IDs2tags(tagIDs)]
                else:
                    raise RuntimeError("Error in function annotate_sentence")

        self.annotate_sentence = annotate_sentence

    def before_run(self):
        self.model.to(self.device)
        logger.info(f"{self.name} using device {next(self.model.parameters()).device}")

    def after_run(self):
        if self.device_on_run:
            self.model.to('cpu')
            torch.cuda.empty_cache()

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


def from_tigertag(tigertag):
    # adapted from https://github.com/rubcompling/konvens2019/scripts/common.py

    def feature(featname, arg):
        value_map = {'Sg': 'Sing', 'Pl': 'Plur'}
        return featname, value_map.get(arg, arg)

    degree = partial(feature, "Degree")
    case = partial(feature, "Case")
    number = partial(feature, "Number")
    gender = partial(feature, "Gender")
    person = partial(feature, "Person")
    tense = partial(feature, "Tense")
    mood = partial(feature, "Mood")

    stts, *parts = tigertag.split(".")
    if stts == '$':
        stts, parts = '$.', []
    _, base_features = conv_table[stts]
    feature_dict = MorphAnalysis(Vocab(), base_features).to_dict()

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

    extracted_fields = dict(f(p) for f, p in zip(fields_for_tag(stts), parts))
    feature_dict.update(extracted_fields)
    return feature_dict
