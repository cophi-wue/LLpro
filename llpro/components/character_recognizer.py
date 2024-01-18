import itertools
import logging
from typing import Callable, List, Iterable

import more_itertools
import spacy
import torch
from spacy import Language
from spacy.tokens import Doc, Span, Token
from tqdm import tqdm
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
    'model': 'aehrm/droc-character-recognizer', 'batch_size': 8, 'use_cuda': True, 'device_on_run': True, 'pbar_opts': None
})
def character_recognizer(nlp, name, model, batch_size, use_cuda, device_on_run, pbar_opts):
    if not Token.has_extension('character_iob'):
        Token.set_extension('character_iob', default='O')
    if not Doc.has_extension('characters'):
        Doc.set_extension('characters', getter=calc_character_spans)
    return CharacterRecognizer(name, model, batch_size, use_cuda, device_on_run, pbar_opts)


class CharacterRecognizer(Module):

    def __init__(self, name, model='aehrm/droc-character-recognizer', batch_size=8, use_cuda=True, device_on_run=True, pbar_opts=None):
        super().__init__(name, pbar_opts=pbar_opts)
        self.device_on_run = device_on_run
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else "cpu")

        import flair
        from flair.models import SequenceTagger
        from flair.data import Sentence
        from flair.datasets import DataLoader
        from flair.datasets import FlairDatapointDataset
        # see https://github.com/flairNLP/flair/issues/2650#issuecomment-1063785119
        flair.device = 'cpu'
        self.tagger = SequenceTagger.load(model)

        def process_batch(sentence_batch: Iterable[spacy.tokens.Span]) -> List[flair.data.Span]:
            flair_sentences = [Sentence([tok.text for tok in s]) for s in sentence_batch]
            dataloader = DataLoader(
                dataset=FlairDatapointDataset(flair_sentences),
                batch_size=len(flair_sentences),
            )

            output = []
            with torch.no_grad():
                for batch in dataloader:
                    self._annotate_batch(batch)
                    for s in batch:
                        s.clear_embeddings()
                        output.append(s)
            torch.cuda.empty_cache()
            return output

        self.batch_processor = process_batch

        if not self.device_on_run:
            self.tagger.to(self.device)
            logger.info(f"{self.name} using device {str(next(self.tagger.parameters()).device)}")

    def before_run(self):
        import flair
        flair.device = self.device
        self.tagger.to(self.device)
        logger.info(f"{self.name} using device {str(next(self.tagger.parameters()).device)}")

    def after_run(self):
        import flair
        if self.device_on_run:
            self.tagger.to('cpu')
            flair.device = 'cpu'
            torch.cuda.empty_cache()

    def _annotate_batch(self, batch):
        from flair.data import get_spans_from_bio
        import torch

        # cf. flair/models/sequence_tagger_model.py
        # get features from forward propagation
        sentence_tensor, lengths = self.tagger._prepare_tensors(batch)
        features = self.tagger.forward(sentence_tensor, lengths)

        # make predictions
        if self.tagger.use_crf:
            predictions, all_tags = self.tagger.viterbi_decoder.decode(features, False)
        else:
            predictions, all_tags = self.tagger._standard_inference(features, batch, False)

        for sentence, sentence_predictions in zip(batch, predictions):
            if self.tagger.predict_spans:
                sentence_tags = [label[0] for label in sentence_predictions]
                sentence_scores = [label[1] for label in sentence_predictions]
                predicted_spans = get_spans_from_bio(sentence_tags, sentence_scores)
                for predicted_span in predicted_spans:
                    span = sentence[predicted_span[0][0]:predicted_span[0][-1] + 1]
                    span.add_label(self.tagger.tag_type, value=predicted_span[2], score=predicted_span[1])

            # token-labels can be added directly ("O" and legacy "_" predictions are skipped)
            else:
                for token, label in zip(sentence.tokens, sentence_predictions):
                    if label[0] in ["O", "_"]:
                        continue
                    token.add_label(typename=self.tagger.tag_type, value=label[0], score=label[1])

    def process(self, doc: Doc, pbar: tqdm) -> Doc:
        sentences = list(doc.sents)
        mentions = []
        tagged_sentences = itertools.chain.from_iterable(
            self.batch_processor(batch) for batch in more_itertools.chunked(sentences, n=self.batch_size))
        for sentence, tagged_sentence in zip(sentences, tagged_sentences):
            if self.tagger.predict_spans:
                for span in tagged_sentence.get_spans('character'):
                    new_span = sentence[span[0].idx - 1:span[-1].idx]
                    new_span = Span(doc, new_span.start, new_span.end, label=span.tag)
                    mentions.append(new_span)
            else:
                for tok, tagged_tok in zip(sentence, tagged_sentence):
                    value = tagged_tok.get_label('character').value
                    if value in {'O', '_'}: continue
                    new_span = Span(doc, tok.i, tok.i + 1, label=value)
                    mentions.append(new_span)
            pbar.update(len(sentence))

        for mention in mentions:
            for i in range(mention.start, mention.end):
                if i == mention.start:
                    doc[i]._.character_iob = 'B'
                else:
                    doc[i]._.character_iob = 'I'
        return doc

