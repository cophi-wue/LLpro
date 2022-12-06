from pathlib import Path

from someweta import ASPTagger
from spacy import Language
from spacy.morphology import Morphology
from spacy.tokens import Doc, MorphAnalysis
from typing import Callable

from ..stts2upos import conv_table
from ..common import Module


@Language.factory("tagger_someweta", assigns=['token.pos', 'token.tag', 'token.morph'], default_config={'model': 'resources/german_newspaper_2020-05-28.model', 'pbar_opts': None})
def tagger_someweta(nlp, name, model, pbar_opts):
    return SoMeWeTaTagger(name=name, model=model, pbar_opts=pbar_opts)


class SoMeWeTaTagger(Module):

    def __init__(self, name, model='resources/german_newspaper_2020-05-28.model', pbar_opts=None):
        super().__init__(name, pbar_opts=pbar_opts)
        self.tagger = ASPTagger()
        self.tagger.load(str(Path(model)))

    def process(self, doc: Doc, progress_fn: Callable[[int], None]) -> Doc:
        for sentence in doc.sents:
            tagged = self.tagger.tag_sentence([tok.text for tok in sentence])
            assert len(sentence) == len(tagged)
            for token, (tok, tag) in zip(sentence, tagged):
                assert token.text == tok
                upos, tagged_morph = conv_table[tag]
                token.pos_ = upos
                token.tag_ = tag

                morph = token.morph.to_dict()
                morph.update(Morphology.feats_to_dict(tagged_morph))
                token.set_morph(MorphAnalysis(token.vocab, morph))
                progress_fn(1)

        return doc
