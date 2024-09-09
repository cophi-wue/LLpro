import itertools
import logging
import queue
import sys
from pathlib import Path

import more_itertools
import multiprocessing_on_dill as multiprocessing
from spacy import Language
from spacy.tokens import Span, Doc
from typing import List, Iterable, Callable, Tuple, Sequence, Union, Iterator

from tqdm import tqdm

from ..spacy_cython_utils import apply_dependency_to_doc
from ..common import Module
from .. import LLPRO_RESOURCES_ROOT, LLPRO_TEMPDIR

logger = logging.getLogger(__name__)


def get_noun_chunks_parzu(doclike: Union[Doc,Span]) -> Iterator[Tuple[int, int, int]]:
    # TODO Very tentative placeholder. I am not a linguist :(
    # cf. https://github.com/explosion/spaCy/blob/master/spacy/lang/de/syntax_iterators.py
    doc = doclike.doc  # Ensure works on both Doc and Span.
    np_label = doc.vocab.strings.add("NP")
    rbracket = 0
    prev_end = -1
    for i, word in enumerate(doclike):
        if i < rbracket:
            continue
        if word.left_edge.i <= prev_end:
            continue
        if word.pos_ in ["NN", "NE", "PPER"]:
            rbracket = word.i + 1
            # try to extend the span to the right
            # to capture close apposition/measurement constructions
            for rdep in doc[word.i].rights:
                if rdep.pos_ in ["NN", "NE"] and rdep.dep_ == 'app':
                    rbracket = rdep.i + 1
            prev_end = rbracket - 1
            yield word.left_edge.i, rbracket, np_label



@Language.factory("parser_parzu", requires=['token.tag'], assigns=['token.dep', 'token.head'],
                  default_config={
                      'parzu_home': LLPRO_RESOURCES_ROOT + '/ParZu',
                      'parzu_tmpdir': LLPRO_TEMPDIR,
                      'zmorge_transducer': LLPRO_RESOURCES_ROOT + '/zmorge-20150315-smor_newlemma.ca',
                      'pbar_opts': None
                  })
def parser_parzu(nlp, name, parzu_home, parzu_tmpdir, zmorge_transducer, pbar_opts):
    nlp.vocab.get_noun_chunks = get_noun_chunks_parzu
    return ParzuParser(name=name, parzu_home=parzu_home, parzu_tmpdir=parzu_tmpdir,
                             zmorge_transducer=zmorge_transducer, pbar_opts=pbar_opts)


class ParzuParser(Module):
    """
    NOTE: This parser does not assign any head or dep attributes to punctuation signs. Otherwise, Spacy's sentence
    segmentation fails to generate proper sentences, therefore we restitute the sentence segmentation present prior
    to the parsing step. Observe that punctuation is not explicitly handled in Foth's Dependency Grammar,
    hence we do not lose any information.
    """

    def __init__(self, name, parzu_home=LLPRO_RESOURCES_ROOT + '/ParZu',
                 parzu_tmpdir=LLPRO_TEMPDIR,
                 zmorge_transducer=LLPRO_RESOURCES_ROOT + '/zmorge-20150315-smor_newlemma.ca',
                 pbar_opts=None):
        super().__init__(name, pbar_opts=pbar_opts)
        sys.path.insert(0, str(parzu_home))
        from parzu_class import process_arguments, Parser

        self.opts = process_arguments(commandline=False)
        self.opts['smor_model'] = zmorge_transducer
        self.opts['tempdir'] = parzu_tmpdir
        self.parser = Parser(self.opts, timeout=1000)


    def process(self, doc: Doc, pbar: tqdm) -> Doc:
        results = []
        for sent in doc.sents:
            serialized_sent = [(tok.i, tok.text, tok.tag_) for tok in sent]
            result = self.parse_sentence(serialized_sent)
            pbar.update(len(serialized_sent))
            results.extend(result)

        # we use the cython implementation here since, when assigning head and deprel to each token individually in
        # python, this gets extremely slow since for each token, assigning the head attribute triggers recalculations
        # of left/rightmost children. Also, spacy's implementation messes up the sentence boundaries.
        doc = apply_dependency_to_doc(doc, [token_result['head'] for token_result in results], [token_result['deprel'] for token_result in results])
        return doc


    def parse_sentence(self, serialized_sentence: List[Tuple]):
        newinput = []
        for _, text, tag in serialized_sentence:
            newinput.append(text + '\t' + tag)

        reformatted_input = "\n".join(newinput)

        output = self.parser.main(
            reformatted_input, inputformat="tagged", outputformat="conll"
        )

        index_of_first_token, _, _ = serialized_sentence[0]
        it = iter(serialized_sentence)
        for line in output[0].split('\n'):
            if line.strip() == '':
                continue
            i, _, _ = next(it)
            (
                _,
                word,
                lemma,
                _,
                pos,
                feats,
                head,
                deprel,
                deps,
                misc,
            ) = line.strip().split("\t")
            if deprel == 'root':
                head = None
            else:
                head = int(line.split('\t')[6])

            yield {'index': i,
                   'head': i if head is None else index_of_first_token + int(head) - 1,
                   'deprel': deprel}
