import logging
from typing import Callable

import more_itertools
import numpy as np
import torch
from spacy import Language
from spacy.tokens import Doc, Token, Span
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel, BertTokenizer, DataCollatorForTokenClassification, \
    BertTokenizerFast
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.pipelines.base import Dataset

from ..common import Module

logger = logging.getLogger(__name__)

@Language.factory("scene_segmenter", assigns=['doc._.scenes', 'token._.scene'], default_config={
    'model': 'aehrm/stss-scene-segmenter', 'use_cuda': True, 'device_on_run': True,
    'pbar_opts': None
})
def scene_segmenter(nlp, name, model, use_cuda, device_on_run, pbar_opts):
    if not Doc.has_extension('scenes'):
        Doc.set_extension('scenes', default=list())
    if not Token.has_extension('scene'):
        Token.set_extension('scene', default=None)
    return SceneSegmenter(name, model=model, use_cuda=use_cuda, device_on_run=device_on_run, pbar_opts=pbar_opts)

class SceneSegmenter(Module):

    def __init__(self, name, model='aehrm/stss-scene-segmenter', use_cuda=True,
                 device_on_run=True, pbar_opts=None):
        super().__init__(name, pbar_opts=pbar_opts)
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else "cpu")
        self.device_on_run = device_on_run
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.batch_size = 8
        self.model = BertForSentTransitions.from_pretrained(model)
        self.model.eval()
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
        # Joining the words with ' ' should be a sufficiently good approximation of the original text,
        # as the text is further tokenized by BERT in the scene segmenter anyway, which always splits at whitespace.
        sentences = [' '.join(tok.text for tok in sent) for sent in doc.sents]

        def truncated_sents():
            for sent in sentences:
                tokenized = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sent))
                # limit each sentence to 300 subword tokens
                yield tokenized[:300]

        # limit each of BERT's input to 500 subword tokens, but at most 20 sentences
        for chunk in more_itertools.constrained_batches(truncated_sents(), max_size=500, get_len=lambda x: len(x) + 2,
                                                        max_count=20):
            in_seq = [self.tokenizer.cls_token_id]
            for sent in chunk:
                in_seq.extend(sent + [self.tokenizer.sep_token_id])

            out = self.tokenizer(self.tokenizer.decode(in_seq), add_special_tokens=False)
            yield out

    def process(self, doc: Doc, progress_fn: Callable[[int], None]) -> Doc:
        inputs = self.input_gen(doc)
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        sentences = list(doc.sents)
        sentence_counter = 0
        predictions = []
        with torch.no_grad():
            for batch in more_itertools.batched(inputs, self.batch_size):
                out = self.model(**data_collator(batch).to(self.device))
                pred = np.argmax(out.logits.cpu().numpy(), axis=1)
                for p in pred:
                    predictions.append(p)
                    progress_fn(len(sentences[sentence_counter]))
                    sentence_counter = sentence_counter + 1

        assert len(predictions) == len(sentences)
        decoded = self.model.decode_labels(predictions)


        for segment_counter, span in enumerate(more_itertools.split_before(zip(sentences, decoded), lambda x: x[1].endswith('-B'))):
            span_sents, span_labels = zip(*span)
            label = span_labels[0].replace('-B', '')
            scene_obj = Span(doc=doc, start=span_sents[0].start, end=span_sents[-1].end, span_id=segment_counter, label=label)
            doc._.scenes.append(scene_obj)

            for tok in scene_obj:
                tok._.scene = scene_obj



class BertForSentTransitions(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForSentTransitions, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        sentences_mask = input_ids == self.config.sep_token_id
        if labels is not None:
            assert torch.all((labels != -100) == sentences_mask)

        out = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sep_states = out.last_hidden_state[sentences_mask]
        sep_states = self.dropout(sep_states)
        logits = torch.zeros(*input_ids.shape, self.config.num_labels).to(input_ids.device)
        logits[sentences_mask] = self.classifier(sep_states)
        logits[~sentences_mask] = float('nan')
        logits_masked = logits[sentences_mask]

        if labels is not None:
            labels_masked = labels[sentences_mask]
            assert torch.all((labels_masked >= 0) & (labels_masked < self.config.num_labels))
            loss_fct = CrossEntropyLoss(weight=torch.tensor([45,0.1,1500,58]).float().to(input_ids.device))
            loss = loss_fct(logits_masked, labels_masked)
            return TokenClassifierOutput(loss=loss, logits=logits)
        else:
            return TokenClassifierOutput(logits=logits_masked)

    def decode_labels(self, predictions, as_spans=False):
        labels_str = [ self.config.id2label[i] for i in predictions ]
        decoded = []
        for l in labels_str:
            if len(decoded) == 0:
                if not l.endswith('-B'):
                    l = l + '-B'
                decoded.append(l)
            else:
                if l.endswith('-B'):
                    if l == 'Nonscene-B' and decoded[-1] == 'Nonscene':
                        decoded.append('Nonscene')
                    else:
                        decoded.append(l)
                else:
                    if l == decoded[-1].replace('-B', ''):
                        decoded.append(l)
                    else:
                        decoded.append(l + '-B')

        if not as_spans:
            return decoded



