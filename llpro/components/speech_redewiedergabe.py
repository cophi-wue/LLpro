import itertools
import logging
from pathlib import Path

from typing import Dict, Callable

import more_itertools
import torch
from spacy import Language
from spacy.tokens import Doc, Token
from tqdm import tqdm

from ..common import Module
from .. import LLPRO_RESOURCES_ROOT

logger = logging.getLogger(__name__)


@Language.factory("speech_redewiedergabe", assigns=['token._.speech', 'token._.speech_prob'], default_config={
    'models': None, 'batch_size': 8, 'use_cuda': True, 'device_on_run': True, 'pbar_opts': None
})
def speech_redewiedergabe(nlp, name, models, batch_size, use_cuda, device_on_run, pbar_opts):
    if not Token.has_extension('speech'):
        Token.set_extension('speech', default=list())
    if not Token.has_extension('speech_prob'):
        Token.set_extension('speech_prob', default=dict())
    return RedewiedergabeTagger(name=name, models=models, batch_size=batch_size, use_cuda=use_cuda, device_on_run=device_on_run,
                                pbar_opts=pbar_opts)


class RedewiedergabeTagger(Module):

    def __init__(self, name, models=None, batch_size=8, use_cuda=True, device_on_run=True, pbar_opts=None):
        super().__init__(name, pbar_opts=pbar_opts)
        import torch
        import flair
        from flair.models import SequenceTagger
        flair.device = 'cpu'

        self.model_paths = models if models is not None else \
            {'direct': 'aehrm/redewiedergabe-direct',
             'indirect': 'aehrm/redewiedergabe-indirect',
             'reported': 'aehrm/redewiedergabe-reported',
             'freeIndirect': 'aehrm/redewiedergabe-freeindirect'}
        self.device_on_run = device_on_run
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else "cpu")
        self.batch_size = batch_size

        self.models: Dict[str, SequenceTagger] = {}
        for rw_type, model_path in self.model_paths.items():
            model = SequenceTagger.load(model_path)
            model = model.eval()
            self.models[rw_type] = model

        if not self.device_on_run:
            for rw_type, model in self.models.items():
                self.set_model_device(model, rw_type)

    def set_model_device(self, model, rw_type):
        import flair
        flair.device = self.device
        model.to(self.device)
        logger.info(
            f"{self.name}/{rw_type} using device {str(next(model.parameters()).device)}")

    def reset_model_device(self):
        import flair
        flair.device = 'cpu'
        for model in self.models.values():
            model.to('cpu')
        torch.cuda.empty_cache()

    def process(self, doc: Doc, pbar: tqdm) -> Doc:
        from flair.data import Sentence
        max_seq_length = 300  # this a constant in the model

        def gen_sentences():
            for sent  in doc.sents:
                tokenized = [x.text for x in sent]
                yield from more_itertools.chunked(tokenized, n=max_seq_length)

        def gen_inputseq():
            for list_of_sentences in more_itertools.constrained_batches(gen_sentences(), max_size=max_seq_length):# get_len=lambda x: self.bert_sequence_length(x)):
                yield itertools.chain(*list_of_sentences)

        for rw_type, model in self.models.items():
            pbar.reset()
            pbar.total = len(doc)
            pbar.set_description(rw_type)

            if self.device_on_run:
                self.set_model_device(model, rw_type)

            it = iter(doc)
            for chunk in more_itertools.chunked(gen_inputseq(), n=self.batch_size):
                chunk = [list(sent) for sent in chunk]
                chunk_tokens = list(itertools.islice(it, sum(len(x) for x in chunk)))

                chunk_token_it = iter(chunk_tokens)
                sent_objs = [Sentence(sent) for sent in chunk]

                with torch.no_grad():
                    model.predict(sent_objs)

                for sent in sent_objs:
                    for tok_flair in sent:
                        tok = next(chunk_token_it)
                        assert tok_flair.text == tok.text
                        if tok_flair.get_label('cat').to_dict()['value'] != 'O':
                            tok._.speech.append(rw_type)
                        tok._.speech_prob[rw_type] = tok_flair.get_label('cat').to_dict()['confidence']


                pbar.update(sum(len(sent) for sent in chunk))

            if self.device_on_run:
                self.reset_model_device()

        return doc

    def bert_sequence_length(self, seq):
        from flair.embeddings import BertEmbeddings

        for model in self.models.values():
            if type(model.embeddings) == BertEmbeddings:
                return sum(len(model.embeddings.tokenizer.tokenize(x)) for x in seq)
        return len(seq)
