import itertools
import logging
from pathlib import Path

from typing import Dict, Callable

import more_itertools
import numpy as np
import torch
from flair.data import get_spans_from_bio
from spacy import Language
from spacy.tokens import Doc, Token, Span
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

from ..common import Module
from .. import LLPRO_RESOURCES_ROOT

logger = logging.getLogger(__name__)


@Language.factory("speech_redewiedergabe", assigns=['token._.speeches', 'doc._.speeches'], default_config={
    'model': 'aehrm/moderngbert-redewiedergabe', 'batch_size': 1, 'use_cuda': True, 'device_on_run': True, 'pbar_opts': None
})
def speech_redewiedergabe(nlp, name, model, batch_size, use_cuda, device_on_run, pbar_opts):
    if not Doc.has_extension('speeches'):
        Doc.set_extension('speeches', default=list())
    if not Token.has_extension('speeches'):
        Token.set_extension('speeches', default=list())
    return RedewiedergabeTagger(name=name, model=model, batch_size=batch_size, use_cuda=use_cuda, device_on_run=device_on_run,
                                pbar_opts=pbar_opts)


class RedewiedergabeTagger(Module):

    def __init__(self, name, model='aehrm/moderngbert-redewiedergabe', batch_size=1, use_cuda=True, device_on_run=True, pbar_opts=None):
        super().__init__(name, pbar_opts=pbar_opts)
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else "cpu")
        self.device_on_run = device_on_run
        self.batch_size = batch_size
        self.speech_labels = [
            "direct.speech",
            "direct.thought",
            "direct.writing",
            "indirect.speech",
            "indirect.thought",
            "indirect.writing",
            "freeIndirect.speech",
            "freeIndirect.thought",
            "freeIndirect.writing",
            "reported.speech",
            "reported.thought",
            "reported.writing",
        ]

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForTokenClassification.from_pretrained(model, num_labels=3 * len(self.speech_labels))
        self.model.eval()
        self.max_length = self.tokenizer.model_max_length - 2
        if not self.device_on_run:
            self.model.to(self.device)

    def before_run(self):
        self.model.to(self.device)
        logger.info(f"{self.name} using device {next(self.model.parameters()).device}")

    def after_run(self):
        if self.device_on_run:
            self.model.to('cpu')
            torch.cuda.empty_cache()

    def input_gen(self, doc):
        def tokenize_sentences():
            for sent in doc.sents:
                offset = sent[0].i
                tokenized = self.tokenizer([tok.text for tok in sent], is_split_into_words=True, truncation=True,
                                           add_special_tokens=False, max_length=self.max_length)
                yield tokenized['input_ids'], [offset + i for i in tokenized.word_ids()]

        for chunk in more_itertools.constrained_batches(tokenize_sentences(), max_size=self.max_length,
                                                        get_len=lambda x: len(x[0])):
            input_ids, word_ids = zip(*chunk)
            in_seq = [self.tokenizer.cls_token_id] + list(itertools.chain.from_iterable(input_ids)) + [self.tokenizer.sep_token_id]
            word_ids = [None] + list(itertools.chain.from_iterable(word_ids)) + [None]
            yield in_seq, word_ids


    def process(self, doc: Doc, pbar: tqdm) -> Doc:
        inputs = self.input_gen(doc)
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        predictions = []
        with torch.no_grad():
            for batch in more_itertools.batched(inputs, self.batch_size):
                input_ids, word_ids = zip(*batch)
                out = self.model(**data_collator([{'input_ids': x} for x in input_ids]).to(self.device))

                batch_size, seq_len, _ = out.logits.shape
                logits = out.logits.reshape(batch_size, seq_len, len(self.speech_labels), 3)
                preds = logits.argmax(-1)

                for i in range(batch_size):
                    subword_mask = []
                    prev_word_id = None
                    for k, word_id in enumerate(word_ids[i]):
                        if word_id is not None and word_id != prev_word_id:
                            subword_mask.append(k)
                        prev_word_id = word_id

                    for j, speech_label in enumerate(self.speech_labels):
                        pred = ['OBI'[x]+'-' for x in preds[i,subword_mask,j]]
                        for bio_span in get_spans_from_bio(pred):
                            token_ids = np.array(word_ids[i])[subword_mask][bio_span[0]]
                            predictions.append((token_ids, speech_label))

                pbar.update(max(x for x in word_ids[-1] if x is not None)-pbar.n)


        i = 0
        for predicted_span in predictions:
            tokens = [doc[tok_id] for tok_id in predicted_span[0]]
            speech = Span(doc=doc, start=tokens[0].i, end=tokens[-1].i+1, span_id=i, label=predicted_span[1])
            for token in tokens:
                token._.speeches.append(speech)

            doc._.speeches.append(speech)
            i = i + 1

        return doc

