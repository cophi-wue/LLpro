import itertools
import logging
from pathlib import Path

import flair
from typing import Dict

import more_itertools
import torch
from spacy import Language
from spacy.tokens import Doc, Token


@Language.factory("speech_redewiedergabe", assigns=['token._.speech', 'token._.speech_prob'], default_config={'model_paths': None})
def speech_redewiedergabe(nlp, name, model_paths):
    if not Token.has_extension('speech'):
        Token.set_extension('speech', default=list())
    if not Token.has_extension('speech_prob'):
        Token.set_extension('speech_prob', default=dict())
    return RedewiedergabeTagger(name=name, model_paths=model_paths)


class RedewiedergabeTagger:

    def __init__(self, name, model_paths=None, use_cuda=True, device_on_run=False):
        import torch
        from flair.models import SequenceTagger
        flair.device = 'cpu'

        self.model_paths = model_paths if model_paths is not None else \
            {'direct': 'resources/rwtagger_models/models/direct/final-model.pt',
             'indirect': 'resources/rwtagger_models/models/indirect/final-model.pt',
             'reported': 'resources/rwtagger_models/models/reported/final-model.pt',
             'freeIndirect': 'resources/rwtagger_models/models/freeIndirect/final-model.pt'}
        self.device_on_run = device_on_run
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else "cpu")

        self.models: Dict[str, SequenceTagger] = {}
        for rw_type, model_path in self.model_paths.items():
            model = SequenceTagger.load(str(Path(model_path)))
            model = model.eval()
            self.models[rw_type] = model

        # if not self.device_on_run:
        #     for model in self.models.values():
        #         model.to(self.device)
        #     logging.info(
        #         f"{name} using devices {','.join(str(next(m.parameters()).device) for m in self.models.values())}")

    # def before_run(self):
    #     flair.device = self.device
    #     if self.device_on_run:
    #         for model in self.models.values():
    #             model.to(self.device)
    #         logging.info(
    #             f"{self.name} using devices {','.join(str(next(m.parameters()).device) for m in self.models.values())}")
    #
    # def after_run(self):
    #     if self.device_on_run:
    #         for model in self.models.values():
    #             model.to('cpu')
    #         torch.cuda.empty_cache()

    def __call__(self, doc: Doc) -> Doc:
        from flair.data import Sentence
        max_seq_length = 510  # inc. [CLS] and [SEP]
        it = iter(doc)

        def gen_sentences():
            for sentence in doc.sents:
                for sent_part in more_itertools.constrained_batches([x.text for x in sentence], max_size=max_seq_length,
                                                                    get_len=lambda x: self.bert_sequence_length(x)):
                    yield sent_part

        for chunk in more_itertools.chunked(gen_sentences(), n=10):
            chunk = [list(sent) for sent in chunk]
            tokens = itertools.islice(it, sum(len(x) for x in chunk))

            for rw_type, model in self.models.items():
                sent_objs = [Sentence(sent) for sent in chunk]
                with torch.no_grad():
                    model.predict(sent_objs)

                labels = itertools.chain.from_iterable(sent.to_dict('cat')['cat'] for sent in sent_objs)
                for tok, label in zip(tokens, labels):
                    if label['value'] != 'x':
                        tok._.speech.append(rw_type)
                        tok._.speech_prob[rw_type] = label['confidence']

            # update_fn(sum(len(sent) for sent in chunk))
        return doc

    def bert_sequence_length(self, seq):
        from flair.embeddings import BertEmbeddings

        for model in self.models.values():
            if type(model.embeddings) == BertEmbeddings:
                return sum(len(model.embeddings.tokenizer.tokenize(x)) for x in seq)
        return len(seq)
