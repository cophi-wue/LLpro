import itertools
import logging
from pathlib import Path

from typing import Dict, Callable

import more_itertools
import torch
from spacy import Language
from spacy.tokens import Doc, Token

from ..common import Module
from .. import LLPRO_RESOURCES_ROOT

logger = logging.getLogger(__name__)


@Language.factory("speech_redewiedergabe", assigns=['token._.speech', 'token._.speech_prob'], default_config={
    'model_paths': None, 'use_cuda': True, 'device_on_run': True, 'pbar_opts': None
})
def speech_redewiedergabe(nlp, name, model_paths, use_cuda, device_on_run, pbar_opts):
    if not Token.has_extension('speech'):
        Token.set_extension('speech', default=list())
    if not Token.has_extension('speech_prob'):
        Token.set_extension('speech_prob', default=dict())
    return RedewiedergabeTagger(name=name, model_paths=model_paths, use_cuda=use_cuda, device_on_run=device_on_run,
                                pbar_opts=pbar_opts)


class RedewiedergabeTagger(Module):

    def __init__(self, name, model_paths=None, use_cuda=True, device_on_run=True, pbar_opts=None):
        super().__init__(name, pbar_opts=pbar_opts)
        import torch
        import flair
        from flair.models import SequenceTagger
        flair.device = 'cpu'

        # TODO upload custom models
        self.model_paths = model_paths if model_paths is not None else \
            {'direct': LLPRO_RESOURCES_ROOT + '/rwtagger_models/models/direct/final-model.pt',
             'indirect': LLPRO_RESOURCES_ROOT + '/rwtagger_models/models/indirect/final-model.pt',
             'reported': LLPRO_RESOURCES_ROOT + '/rwtagger_models/models/reported/final-model.pt',
             'freeIndirect': LLPRO_RESOURCES_ROOT + '/rwtagger_models/models/freeIndirect/final-model.pt'}
        self.device_on_run = device_on_run
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else "cpu")

        self.models: Dict[str, SequenceTagger] = {}
        for rw_type, model_path in self.model_paths.items():
            model = SequenceTagger.load(str(Path(model_path)))
            model = model.eval()
            self.models[rw_type] = model

        if not self.device_on_run:
            for model in self.models.values():
                model.to(self.device)
            flair.device = self.device
            logger.info(
                f"{name} using devices {','.join(str(next(m.parameters()).device) for m in self.models.values())}")

    def before_run(self):
        import flair
        flair.device = self.device
        for model in self.models.values():
            model.to(self.device)
        logger.info(
            f"{self.name} using devices {','.join(str(next(m.parameters()).device) for m in self.models.values())}")

    def after_run(self):
        if self.device_on_run:
            import flair
            flair.device = 'cpu'
            for model in self.models.values():
                model.to('cpu')
            torch.cuda.empty_cache()

    def process(self, doc: Doc, progress_fn: Callable[[int], None]) -> Doc:
        from flair.data import Sentence
        max_seq_length = 100  # inc. [CLS] and [SEP]
        it = iter(doc)

        def gen_sentences():
            tokenized_sentences = ([x.text for x in sentence] for sentence in doc.sents)
            for list_of_sentences in more_itertools.constrained_batches(tokenized_sentences, max_size=max_seq_length):# get_len=lambda x: self.bert_sequence_length(x)):
                yield itertools.chain(*list_of_sentences)

        for chunk in more_itertools.chunked(gen_sentences(), n=10):
            chunk = [list(sent) for sent in chunk]
            tokens = list(itertools.islice(it, sum(len(x) for x in chunk)))

            for rw_type, model in self.models.items():
                sent_objs = [Sentence(sent) for sent in chunk]
                with torch.no_grad():
                    model.predict(sent_objs)

                labels = itertools.chain.from_iterable(sent.to_dict('cat')['cat'] for sent in sent_objs)
                for tok, label in zip(tokens, labels):
                    # label = label[0].to_dict()
                    if label['value'] != 'x':
                        tok._.speech.append(rw_type)
                    tok._.speech_prob[rw_type] = label['confidence']

            progress_fn(sum(len(sent) for sent in chunk))
        return doc

    def bert_sequence_length(self, seq):
        from flair.embeddings import BertEmbeddings

        for model in self.models.values():
            if type(model.embeddings) == BertEmbeddings:
                return sum(len(model.embeddings.tokenizer.tokenize(x)) for x in seq)
        return len(seq)
