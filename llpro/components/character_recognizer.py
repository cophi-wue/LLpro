import itertools
import logging
from typing import Callable, List

import more_itertools
import torch
from spacy import Language
from spacy.tokens import Doc, Span, Token
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from ..common import Module

logger = logging.getLogger(__name__)

def calc_character_spans(doc: Doc) -> List[Span]:
    start = -1
    ent_id = 0
    output = []
    for i in range(len(doc)):
        token = doc[i]
        if token._.character_iob == 'I':
            if start == -1:
                # silently ignore that we see O, I-PER opening
                start = i
                ent_id = ent_id
        elif token._.character_iob == 'O':
            if start != -1:
                output.append(Span(doc, start, i, span_id=ent_id))
                ent_id = ent_id + 1
            start = -1
        elif token._.character_iob == 'B':
            if start != -1:
                output.append(Span(doc, start, i, span_id=ent_id))
                ent_id = ent_id + 1
            start = i
    if start != -1:
        output.append(Span(doc, start, doc.length, span_id=ent_id))
    return output

@Language.factory("character_recognizer", assigns=['doc._.characters', 'token._.character_iob'], default_config={
    'batch_size': 8, 'use_cuda': True, 'device_on_run': True, 'pbar_opts': None
})
def character_recognizer(nlp, name, batch_size, use_cuda, device_on_run, pbar_opts):
    if not Token.has_extension('character_iob'):
        Token.set_extension('character_iob', default='O')
    if not Doc.has_extension('characters'):
        Doc.set_extension('characters', getter=calc_character_spans)
    return CharacterRecognizer(name, batch_size, use_cuda, device_on_run, pbar_opts)


class CharacterRecognizer(Module):

    def __init__(self, name, batch_size=8, use_cuda=True, device_on_run=True, pbar_opts=None):
        super().__init__(name, pbar_opts=pbar_opts)
        self.device_on_run = device_on_run
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained('severinsimmler/literary-german-bert')
        self.model = AutoModelForTokenClassification.from_pretrained('severinsimmler/literary-german-bert')
        # self.pipeline = pipeline('ner', model='severinsimmler/literary-german-bert', aggregation_strategy='first')

    def before_run(self):
        self.model.to(self.device)
        logger.info(f"{self.name} using device {str(next(self.model.parameters()).device)}")

    def after_run(self):
        if self.device_on_run:
            self.model.to('cpu')
            torch.cuda.empty_cache()

    def process(self, doc: Doc, progress_fn: Callable[[int], None]) -> Doc:
        max_seq_length = self.model.config.max_position_embeddings - 2

        it = iter(doc)

        def gen_sentences():
            tokenized_sentences = ([x.text for x in sentence] for sentence in doc.sents)
            for list_of_sentences in more_itertools.constrained_batches(tokenized_sentences, max_size=max_seq_length, get_len=lambda x: self.bert_sequence_length(x)):
                yield itertools.chain(*list_of_sentences)

        for chunk in more_itertools.chunked(gen_sentences(), n=self.batch_size):
            chunk = [list(sent) for sent in chunk]

            inputs = self.tokenizer(chunk, is_split_into_words=True, return_tensors='pt', padding=True, truncation=False)
            with torch.no_grad():
                logits = self.model(**inputs.to(self.device)).logits
                predictions = logits.argmax(axis=-1).tolist()

            for i in range(len(chunk)):
                input_seq = inputs[i]
                pred = predictions[i]
                labels = [self.model.config.id2label[j] for j in pred]
                for word_id in itertools.count(start=0,step=1):
                    word_slice = input_seq.word_to_tokens(word_id)
                    if word_slice is None:
                        break
                    spacy_word = next(it)
                    assert self.tokenizer.convert_tokens_to_string(input_seq.tokens[slice(*word_slice)]).replace(' ', '') == spacy_word.text
                    spacy_word._.character_iob = labels[word_slice[0]].replace('-PER', '')  # take label from first subword token
                    progress_fn(1)

        return doc



    def bert_sequence_length(self, seq):
        return sum(len(self.tokenizer.tokenize(x)) for x in seq)
