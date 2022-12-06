import io
import itertools
import os

import spacy
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, ElectraTokenizer
from transformers.models.bert import BasicTokenizer
from ts.torch_handler.base_handler import BaseHandler

import event_classify.preprocessing
from event_classify.datasets import JSONDataset, SpanAnnotation
from event_classify.eval import evaluate
from event_classify.parser import Parser
from event_classify.preprocessing import build_pipeline
from event_classify.util import get_config, get_model, split_text


class EventHandler(BaseHandler):
    def initialize(self, context):
        self._context = context
        self.initialized = True
        properties = context.system_properties

        self.manifest = context.manifest
        model_dir = properties.get("model_dir")
        self.batch_size = properties.get("batch_size")

        spacy.require_gpu(properties.get("gpu_id"))
        self.nlp = build_pipeline(Parser.SPACY)
        properties = context.system_properties
        self.config = get_config(os.path.join(model_dir, "config"))
        self.device = torch.device(properties.get("gpu_id") if torch.cuda.is_available() else "cpu")
        device_id = properties.get("gpu_id") if torch.cuda.is_available() else None
        self.model, self.tokenizer = get_model(os.path.join(model_dir, "model"), config=self.config)
        self.model.to(self.device)


    def preprocess(self, data):
        """
        Transform raw input into model input data.
        """
        # Take the input data and make it inference ready
        inner_data = data[0].get("data")
        if inner_data is None:
            inner_data = data[0].get("body")
        splits = split_text(inner_data.get("text"))
        input_data = {
            "text": inner_data.get("text"),
            "title": None,
            "annotations": []
        }
        for split in splits:
            doc = self.nlp(split.text)
            annotations = event_classify.preprocessing.get_annotation_dicts(doc)
            for annotation in annotations:
                annotation["start"] += split.offset
                annotation["end"] += split.offset
                new_spans = []
                for span in annotation["spans"]:
                    new_spans.append((
                        span[0] + split.offset,
                        span[1] + split.offset,
                    ))
                annotation["spans"] = new_spans
            input_data["annotations"].extend(annotations)
        return input_data


    def postprocess(self, inference_output):
        assert len(inference_output) == 2
        dataset, evaluation_result = inference_output
        try:
            out_data = dataset.get_annotation_json(evaluation_result)[0]
        except IndexError:
            out_data = {
                "annotations": [],
                "text": "",
            }
        return [out_data]

    def inference(self, data, *args, **kwargs):
        dataset = JSONDataset(dataset_file=None, data=[data], include_special_tokens=self.config.dataset.special_tokens)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=lambda list_: SpanAnnotation.to_batch(list_, self.tokenizer),
        )
        evaluation_result = evaluate(loader, self.model, device=self.device)
        return dataset, evaluation_result
