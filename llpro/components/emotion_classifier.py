import itertools
import logging
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple, List

import more_itertools
import torch
from spacy import Language
from spacy.tokens import Doc, Span, Token
from transformers import BertForSequenceClassification, BertTokenizer

from ..common import Module

from .. import LLPRO_RESOURCES_ROOT

logger = logging.getLogger(__name__)

# cf. https://github.com/LeKonArD/Gattungen_und_Emotionen_dhd2023/blob/main/predict_shaver.py
def load_checkpoint_cls(model_name, path):
    model = BertForSequenceClassification.from_pretrained(model_name)
    state_dict = torch.load(path, map_location='cpu')

    mkeys = list(model.state_dict().keys())
    skeys = list(state_dict.keys())

    for k in skeys:

        if k not in mkeys:
            del state_dict[k]

    mkeys = list(model.state_dict().keys())
    skeys = list(state_dict.keys())

    for k in mkeys:

        if k not in skeys:
            state_dict[k] = model.state_dict()[k]

    model.load_state_dict(state_dict)

    return model

@Language.factory("emotion_classifier", assigns=['doc._.emotions', 'token._.emotions'], default_config={
    'base_model': 'deepset/gbert-large', 'batch_size': 8, 'use_cuda': True, 'device_on_run': True, 'pbar_opts': None,
    'weights_dir': LLPRO_RESOURCES_ROOT + '/konle_emotion_weights'
})
def emotion_classifier(nlp, name, base_model, weights_dir, batch_size, use_cuda, device_on_run, pbar_opts):
    if not Token.has_extension('emotions'):
        Token.set_extension('emotions', default=list())
    if not Doc.has_extension('emotions'):
        Doc.set_extension('emotions', default=list())
    return EmotionClassifier(name=name, base_model=base_model, weights_dir=weights_dir, batch_size=batch_size, use_cuda=use_cuda, device_on_run=device_on_run,
                                pbar_opts=pbar_opts)

class EmotionClassifier(Module):
    def __init__(self, name, base_model='deepset/gbert-large', weights_dir=LLPRO_RESOURCES_ROOT + '/konle_emotion_weights', batch_size=8, use_cuda=True, device_on_run=True, pbar_opts=None):
        super().__init__(name, pbar_opts=pbar_opts)
        self.device_on_run = device_on_run
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else "cpu")
        self.base_model = base_model
        self.weights_dir = Path(weights_dir)

        self.tokenizer = BertTokenizer.from_pretrained(base_model)

        self.models: Dict[str, BertForSequenceClassification] = {}
        for emo_type in ['Agitation', 'Anger', 'Fear', 'Joy', 'Love', 'Sadness']:
            logger.info(f"loading {str(self.weights_dir / (emo_type + '.pt'))}")
            self.models[emo_type] = load_checkpoint_cls(self.base_model, str(self.weights_dir / (emo_type + '.pt')))

        if not self.device_on_run:
            for model in self.models.values():
                model.to(self.device)
            logger.info(
                f"{name} using devices {','.join(str(next(m.parameters()).device) for m in self.models.values())}")

    def before_run(self):
        for model in self.models.values():
            model.to(self.device)
        logger.info(
            f"{self.name} using devices {','.join(str(next(m.parameters()).device) for m in self.models.values())}")

    def after_run(self):
        for model in self.models.values():
            model.to('cpu')
        torch.cuda.empty_cache()

    def input_gen(self, doc: Doc) -> Sequence[Tuple[Span, List[int]]]:
        for sent in doc.sents:
            subword_tokens = [self.tokenizer.tokenize(tok.text) for tok in sent]
            subword_ctr = 0
            for chunk in more_itertools.constrained_batches(subword_tokens, max_size=self.tokenizer.max_len_single_sentence, get_len=len, strict=True):
                chunk = list(chunk)
                span_len = len(chunk)
                in_seq = self.tokenizer.convert_tokens_to_ids(itertools.chain(*chunk))
                yield sent[subword_ctr:subword_ctr+span_len], self.tokenizer.decode(in_seq)
                subword_ctr = subword_ctr + span_len


    def process(self, doc: Doc, progress_fn: Callable[[int], None]) -> Doc:
        span_id_ctr = 0
        for chunk in more_itertools.chunked(self.input_gen(doc), n=self.batch_size):
            spans, inputs = zip(*chunk)

            prepared_inputs = self.tokenizer(inputs, padding=True, return_tensors='pt').to(self.device)
            for emo_type, model in self.models.items():
                with torch.no_grad():
                    out = model(**prepared_inputs)
                    label = out['logits'].argmax(axis=1).detach().cpu().numpy()

                for i, span in enumerate(spans):
                    if label[i] == 1:
                        emotion_span_obj = Span(doc=doc, start=span.start, end=span.end, label=emo_type, span_id=span_id_ctr)
                        span_id_ctr = span_id_ctr + 1
                        doc._.emotions.append(emotion_span_obj)
                        for tok in span:
                            tok._.emotions.append(emotion_span_obj)

            progress_fn(sum(len(span) for span in spans))
        return doc