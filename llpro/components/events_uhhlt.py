import collections
import logging
import sys
from pathlib import Path
from typing import Callable

import torch
from spacy import Language
from spacy.tokens import Doc, SpanGroup, Span
from torch.utils.data import DataLoader

from ..common import Module


@Language.factory("events_uhhlt", requires=['token.tag', 'token.dep', 'token.head'], assigns=['Doc._.events'],
                  default_config={
                      'event_classify_home': 'resources/uhh-lt-event-classify',
                      'model_dir': './resources/eventclassifier_model/demo_model',
                      'batch_size': 8,
                      'pbar_opts': None,
                      'use_cuda': True,
                      'device_on_run': True})
def events_uhhlt(nlp, name, event_classify_home, model_dir, batch_size, pbar_opts, use_cuda, device_on_run):
    if not Doc.has_extension('events'):
        Doc.set_extension('events', default=list())
    return EventClassifier(name, event_classify_home=event_classify_home, model_dir=model_dir, batch_size=batch_size,
                           pbar_opts=pbar_opts, use_cuda=use_cuda, device_on_run=device_on_run)


class EventClassifier(Module):

    def __init__(self, name, event_classify_home, model_dir, batch_size=8, use_cuda=True, device_on_run=True,
                 pbar_opts=None):
        super().__init__(name, pbar_opts)
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else "cpu")
        self.event_classify_home = Path(event_classify_home)
        self.device_on_run = device_on_run
        self.batch_size = batch_size
        sys.path.insert(0, str(self.event_classify_home))

        from event_classify.util import get_model
        self.model, self.tokenizer = get_model(model_dir)

        if not self.device_on_run:
            self.model.to(self.device)
            logging.info(f"{self.name} using device {next(self.model.parameters()).device}")

    def before_run(self):
        if self.device_on_run:
            self.model.to(self.device)
            logging.info(f"{self.name} using device {next(self.model.parameters()).device}")

    def after_run(self):
        if self.device_on_run:
            self.model.to('cpu')

    def prepare_data(self, doc):
        from event_classify.segmentations import event_segmentation
        from event_classify.preprocessing import get_annotation_dicts
        from event_classify.datasets import JSONDataset, SpanAnnotation
        # NOTE: this populates the doc._.scenes attribute with (unlabeled) verbal phrases
        segmented_doc = event_segmentation(doc)
        annotations = get_annotation_dicts(segmented_doc)
        data = {"text": doc.text, "title": None, "annotations": annotations}

        dataset = JSONDataset(dataset_file=None, data=[data], include_special_tokens=True)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=lambda list_: SpanAnnotation.to_batch(list_, self.tokenizer),
        )
        return dataset, loader

    def format_predictions(self, predictions, extra_predictions):
        from event_classify.datasets import EventType, SpeechType
        event_types = [EventType(p.item()) for p in predictions]
        iterative = []
        mental = []
        i = 0
        for et in event_types:
            if et != EventType.NON_EVENT:
                iterative.append(bool(extra_predictions["iterative"][i].item()))
                mental.append(bool(extra_predictions["mental"][i].item))
                i += 1
            else:
                iterative.append(None)
                mental.append(None)
        return {
            "event_types": event_types,
            "speech_type": [
                SpeechType(p.item()) for p in extra_predictions["speech_type"]
            ],
            "thought_representation": [
                bool(p.item()) for p in extra_predictions["thought_representation"]
            ],
            "iterative": iterative,
            "mental": mental,
        }

    def process(self, doc: Doc, progress_fn: Callable[[int], None]) -> Doc:
        dataset, dataloader = self.prepare_data(doc)

        progress_counter = 0
        # cf. resources/uhh-lt-event-classify/event_classify/eval.py
        self.model.eval()
        predictions = []
        labled_annotations = []
        all_predictions = collections.defaultdict(list)
        for input_data, gold_labels, annotations in dataloader:
            out = self.model(**input_data.to(self.device))
            for anno, label in zip(annotations, out.event_type.cpu()):
                labled_annotations.append((label, anno))
            predictions.append(out.event_type.cpu())
            for name in ["mental", "iterative"]:
                selector = out.event_type != 0
                all_predictions[name].append(
                    torch.masked_select(getattr(out, name).cpu(), selector.cpu())
                )
            for name in ["speech_type", "thought_representation"]:
                all_predictions[name].append(getattr(out, name).cpu())

            new_progress_counter = doc.char_span(annotations[0].start,
                                                 annotations[-1].end).end  # token index of the last processed span
            progress_fn(new_progress_counter - progress_counter)
            progress_counter = new_progress_counter

        all_predictions = {
            name: torch.cat(values) for name, values in all_predictions.items()
        }
        result = self.format_predictions(torch.cat(predictions).cpu(), all_predictions)

        new_spans = []
        for i, event_span in enumerate(doc._.events):
            attrs = {
                k: v[i].to_string() if hasattr(v[i], "to_string") else v[i]
                for k, v in result.items()
            }
            new_spans.append(SpanGroup(doc, spans=event_span, attrs=attrs))
        doc._.events = new_spans
        return doc
