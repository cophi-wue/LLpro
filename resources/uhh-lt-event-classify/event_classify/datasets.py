from __future__ import annotations

import copy
import fnmatch
import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

import catma_gitlab as catma
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

import event_classify.preprocessing
from event_classify.parser import Parser
from event_classify.util import split_text

from .evaluation_result import EvaluationResult

ALL_ANNOTATION_COLLECTIONS = [
    "Verwandlung_MV",
    "Verwandlung_MW",
    "Effi_Briest_GS",
    "Effi_Briest_MW",
    "Eckbert_AN",
    "Eckbert_MW",
    "Judenbuche_AN",
    "Judenbuche_GS",
    "Krambambuli_GS",
    "Krambambuli_MW",
    "Erdbeben_MW",
    "Erdbeben_GS",
]


def get_annotation_collections(exclude: List[str] = []):
    to_remove = set()
    for pattern in exclude:
        for collection in ALL_ANNOTATION_COLLECTIONS:
            if fnmatch.fnmatch(collection, pattern):
                to_remove.add(collection)
    return list(set(ALL_ANNOTATION_COLLECTIONS) - to_remove), list(to_remove)


@dataclass
class EventClassificationLabels:
    event_type: torch.Tensor
    speech_type: torch.Tensor
    thought_representation: torch.Tensor
    # These two are not defined for non_events
    iterative: torch.Tensor
    mental: torch.Tensor

    def to(self, device):
        new = copy.deepcopy(self)
        for member in new.__dataclass_fields__:
            setattr(new, member, getattr(new, member).to(device))
        return new


class EventType(Enum):
    NON_EVENT = 0
    CHANGE_OF_STATE = 1
    PROCESS = 2
    STATIVE_EVENT = 3

    def to_onehot(self):
        out = torch.zeros(4)
        out[self.value] = 1.0
        return out

    def get_narrativity_ordinal(self):
        if self == EventType.NON_EVENT:
            return 0
        elif self == EventType.STATIVE_EVENT:
            return 1
        elif self == EventType.PROCESS:
            return 2
        elif self == EventType.CHANGE_OF_STATE:
            return 3

    @staticmethod
    def from_tag_name(name: str):
        if name == "non_event":
            return EventType.NON_EVENT
        if name == "change_of_state":
            return EventType.CHANGE_OF_STATE
        if name == "process":
            return EventType.PROCESS
        if name == "stative_event":
            return EventType.STATIVE_EVENT
        raise ValueError(f"Invalid Event variant {name}")

    def get_narrativity_score(self):
        if self == EventType.NON_EVENT:
            return 0
        if self == EventType.CHANGE_OF_STATE:
            return 7
        if self == EventType.PROCESS:
            return 5
        if self == EventType.STATIVE_EVENT:
            return 2
        else:
            raise ValueError("Unknown EventType")

    def to_string(self) -> str:
        if self == EventType.NON_EVENT:
            return "non_event"
        if self == EventType.CHANGE_OF_STATE:
            return "change_of_state"
        if self == EventType.PROCESS:
            return "process"
        if self == EventType.STATIVE_EVENT:
            return "stative_event"
        else:
            raise ValueError("Unknown EventType")


class SpeechType(Enum):
    CHARACTER = 0
    NARRATOR = 1
    NONE = 2

    @staticmethod
    def from_list(in_list: List[str]) -> SpeechType:
        if "character_speech" in in_list:
            return SpeechType.CHARACTER
        elif "narrator_speech" in in_list:
            return SpeechType.NARRATOR
        elif len(in_list) > 0:
            return SpeechType.NONE
        else:
            raise ValueError("RepresentationType not specified")

    def to_string(self) -> str:
        if self == SpeechType.CHARACTER:
            return "character"
        elif self == SpeechType.NARRATOR:
            return "narrator"
        else:
            return "none"

    def to_onehot(self, device="cpu"):
        out = torch.zeros(3, device=device)
        out[self.value] = 1.0
        return out


class SpanAnnotation(NamedTuple):
    text: str
    special_token_text: str
    iterative: Optional[bool]
    speech_type: SpeechType
    thought_representation: bool
    mental: Optional[bool]
    event_type: Optional[EventType]
    document_text: str
    # These annotate the start and end offsets in the document string
    start: int
    end: int
    spans: List[Tuple[int, int]]

    @staticmethod
    def to_batch(data: List[SpanAnnotation], tokenizer: PreTrainedTokenizer):
        encoded = tokenizer.batch_encode_plus(
            [anno.special_token_text for anno in data],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if any(anno.event_type is None for anno in data):
            labels = None
        else:
            labels = EventClassificationLabels(
                event_type=torch.tensor([anno.event_type.value for anno in data]),
                mental=torch.tensor(
                    [
                        anno.mental
                        for anno in data
                        if anno.event_type != EventType.NON_EVENT
                    ],
                    dtype=torch.float,
                ),
                iterative=torch.tensor(
                    [
                        anno.iterative
                        for anno in data
                        if anno.event_type != EventType.NON_EVENT
                    ],
                    dtype=torch.float,
                ),
                speech_type=torch.tensor([anno.speech_type.value for anno in data]),
                thought_representation=torch.tensor(
                    [anno.thought_representation for anno in data], dtype=torch.float
                ),
            )
        return encoded, labels, data

    @staticmethod
    def build_special_token_text(
        annotation: catma.Annotation, document, include_special_tokens: bool = True
    ):
        output = []
        plain_text = document.plain_text
        # Provide prefix context
        previous_end = annotation.start_point - 100
        for selection in merge_direct_neighbors(copy.deepcopy(annotation.selectors)):
            output.append(plain_text[previous_end : selection.start])
            if include_special_tokens:
                output.append(" <SE> ")
            output.append(plain_text[selection.start : selection.end])
            if include_special_tokens:
                output.append(" <EE> ")
            previous_end = selection.end
        # Provide suffix context
        output.append(plain_text[previous_end : previous_end + 100])
        return "".join(output)

    @staticmethod
    def build_special_token_text_from_json(
        annotation: Dict, document: str, include_special_tokens: bool = True
    ):
        selections = [tuple(span) for span in annotation["spans"]]
        output = []
        # Provide prefix context
        try:
            previous_end = max(annotation["start"] - 100, 0)
        except KeyError:
            previous_end = max(annotation["spans"][0][0] - 100, 100)
        for start, end in selections:
            output.append(document[previous_end:start])
            if include_special_tokens:
                output.append(" <SE> ")
            output.append(document[start:end])
            if include_special_tokens:
                output.append(" <EE> ")
            previous_end = end
        # Provide suffix context
        output.append(document[previous_end : previous_end + 100])
        return "".join(output)

    def output_dict(self, predictions):
        return {
            "start": self.start,
            "end": self.end,
            "spans": self.spans,
            "predicted": predictions["event_types"],
            "predicted_score": EventType.from_tag_name(
                predictions["event_types"]
            ).get_narrativity_score(),
            "additional_predictions": predictions,
        }


def simplify_representation(repr_list):
    repr_list = [
        t.replace("_1", "").replace("_2", "_").replace("_3", "") for t in repr_list
    ]
    return repr_list


class PlainTextDataset(Dataset):
    def __init__(self, text: str, spacy_device: str = "cpu", language: str = "de"):
        if spacy_device.startswith("cuda"):
            event_classify.preprocessing.use_gpu()
        nlp = event_classify.preprocessing.build_pipeline(Parser.SPACY, language)
        splits = split_text(text)
        self.all_annotations = []
        # Sanity check, splitting should not change text!
        assert text == "".join(split.text for split in splits)
        for split in splits:
            doc = nlp(split.text)
            annotations = event_classify.preprocessing.get_annotation_dicts(doc)
            for annotation in annotations:
                annotation["start"] += split.offset
                annotation["end"] += split.offset
                new_spans = []
                for span in annotation["spans"]:
                    new_spans.append(
                        (
                            span[0] + split.offset,
                            span[1] + split.offset,
                        )
                    )
                annotation["spans"] = new_spans
            for annotation in annotations:
                self.all_annotations.append(
                    SpanAnnotation(
                        text=text[annotation["start"] : annotation["end"]],
                        special_token_text=SpanAnnotation.build_special_token_text_from_json(
                            annotation,
                            text,
                            include_special_tokens=True,
                        ),
                        iterative=None,
                        speech_type=SpeechType.NONE,
                        thought_representation=False,
                        mental=None,
                        event_type=None,
                        document_text=text,
                        # These annotate the start and end offsets in the document string
                        start=annotation["start"],
                        end=annotation["end"],
                        spans=annotation["spans"],
                    )
                )

    def __getitem__(self, i: int):
        return self.all_annotations[i]

    def __len__(self):
        return len(self.all_annotations)


class SimpleJSONEventDataset(Dataset):
    def __init__(self, path: str, include_special_tokens: bool = True):
        """
        Args:
            Path of the extracted annotation data as downlaoded from here: https://zenodo.org/record/6414926
        """
        super().__init__()
        self.annotations: List[SpanAnnotation] = []
        stats = defaultdict(Counter)
        path = Path(path)
        data = json.load(open(path / "Annotations_EvENT.json"))
        for name, collection in data.items():
            # Only use the gold standard for now
            text = SimpleJSONEventDataset.get_full_text(path, name)
            for annotation in collection["gold_standard"]:
                if len(annotation["properties"].get("mental", [])) > 1:
                    logging.warning(
                        "Ignoring annotation with inconsistent 'mental' property"
                    )
                    continue
                try:
                    special_token_text = (
                        SpanAnnotation.build_special_token_text_from_json(
                            annotation,
                            text,
                            include_special_tokens=include_special_tokens,
                        )
                    )
                    simple_representations = simplify_representation(
                        annotation["properties"]["representation_type"]
                    )
                    speech_type = SpeechType.from_list(simple_representations)
                    thought_representation = (
                        "thought_representation" in simple_representations
                    )
                    event_type = EventType.from_tag_name(annotation["tag"])
                    iterative = annotation["properties"].get("iterative", ["no"]) == [
                        "yes"
                    ]
                    mental = annotation["properties"].get("mental", ["no"]) == ["yes"]
                    if event_type == EventType.NON_EVENT:
                        iterative = None
                        mental = None
                    start_annotation = min([span[0] for span in annotation["spans"]])
                    end_annotation = max([span[1] for span in annotation["spans"]])
                    annotation_text = text[start_annotation:end_annotation]
                    span_anno = SpanAnnotation(
                        text=annotation_text,
                        special_token_text=special_token_text,
                        event_type=event_type,
                        iterative=iterative,
                        speech_type=speech_type,
                        thought_representation=thought_representation,
                        mental=mental,
                        document_text=text,
                        start=start_annotation,
                        end=end_annotation,
                        spans=merge_direct_neighbors_json(
                            copy.deepcopy(annotation["spans"])
                        ),
                    )
                    stats["speech_type"].update([speech_type])
                    stats["event_type"].update([event_type])
                    stats["thought_representation"].update([thought_representation])
                    if event_type != EventType.NON_EVENT:
                        stats["iterative"].update([iterative])
                        stats["mental"].update([mental])
                    self.annotations.append(span_anno)
                except ValueError as e:
                    logging.warning(f"Error parsing span annotation: {e}")
        for field_name, variants in stats.items():
            print(f"=== {field_name}")
            total = sum(variants.values())
            for variant_name, variant_count in variants.items():
                print(f"\tWeight {variant_name}:", variant_count / total)

    def __getitem__(self, i: int):
        return self.annotations[i]

    def __len__(self):
        return len(self.annotations)

    @staticmethod
    def get_full_text(basepath: Path, text_name: str):
        out = []
        plain_texts_path = basepath / "Plain_Texts"
        for file_name in os.listdir(plain_texts_path):
            if file_name.endswith(text_name.replace(" ", "_") + ".txt"):
                text_file = open(plain_texts_path / Path(file_name))
                return "".join(text_file.readlines())


class SimpleEventDataset(Dataset):
    """
    Dataset of all event spans with their features.

    This dataset is generated from the CATMA repository which unfortunatly, for various reasons including privacy, can not simply be released.
    """

    def __init__(
        self,
        project: catma.CatmaProject,
        annotation_collections: Iterable[str] = (),
        include_special_tokens: bool = True,
    ):
        """
        Args:
            project: CatmaProject to load from
            annotation_collections: Iterable of annotation collection names to be included
        """
        super().__init__()
        self.annotations: List[SpanAnnotation] = []
        self.tagset = project.tagset_dict["EvENT-Tagset_3"]
        stats = defaultdict(Counter)
        for collection in [project.ac_dict[coll] for coll in annotation_collections]:
            for annotation in collection.annotations:
                if annotation.tag.name in ["Zweifelsfall", "change_of_episode"]:
                    continue  # We ignore these
                if len(annotation.properties.get("mental", [])) > 1:
                    logging.warning(
                        "Ignoring annotation with inconsistent 'mental' property"
                    )
                    continue
                try:
                    special_token_text = SpanAnnotation.build_special_token_text(
                        annotation,
                        collection.text,
                        include_special_tokens=include_special_tokens,
                    )
                    simple_representations = simplify_representation(
                        annotation.properties["representation_type"]
                    )
                    speech_type = SpeechType.from_list(simple_representations)
                    thought_representation = (
                        "thought_representation" in simple_representations
                    )
                    event_type = EventType.from_tag_name(annotation.tag.name)
                    iterative = annotation.properties.get("iterative", ["no"]) == [
                        "yes"
                    ]
                    mental = annotation.properties.get("mental", ["no"]) == ["yes"]
                    if event_type == EventType.NON_EVENT:
                        iterative = None
                        mental = None
                    span_anno = SpanAnnotation(
                        text=annotation.text,
                        special_token_text=special_token_text,
                        event_type=event_type,
                        iterative=iterative,
                        speech_type=speech_type,
                        thought_representation=thought_representation,
                        mental=mental,
                        document_text=collection.text.plain_text,
                        start=annotation.start_point,
                        end=annotation.end_point,
                        spans=[
                            (s.start, s.end)
                            for s in merge_direct_neighbors(
                                copy.deepcopy(annotation.selectors)
                            )
                        ],
                    )
                    stats["speech_type"].update([speech_type])
                    stats["event_type"].update([event_type])
                    stats["thought_representation"].update([thought_representation])
                    if event_type != EventType.NON_EVENT:
                        stats["iterative"].update([iterative])
                        stats["mental"].update([mental])
                    self.annotations.append(span_anno)
                except ValueError as e:
                    logging.warning(f"Error parsing span annotation: {e}")
        for field_name, variants in stats.items():
            print(f"=== {field_name}")
            total = sum(variants.values())
            for variant_name, variant_count in variants.items():
                print(f"\tWeight {variant_name}:", variant_count / total)

    def __getitem__(self, i: int):
        return self.annotations[i]

    def __len__(self):
        return len(self.annotations)


class JSONDataset(Dataset):
    """
    Dataset based on JSON file created by our preprocessing script
    """

    def __init__(
        self,
        dataset_file: Optional[str],
        data: Optional[list] = None,
        include_special_tokens: bool = True,
    ):
        """
        Args:
            dataset_file: Path to json file created by preprocessing script
            data: Instead of a file path read data from this dict instead
        """
        super().__init__()
        self.annotations: List[SpanAnnotation] = []
        self.documents: defaultdict[str, List[SpanAnnotation]] = defaultdict(list)
        if data is None:
            if dataset_file is None:
                raise ValueError("Only one of dataset_file and data may be None")
            else:
                data = json.load(open(dataset_file))
        for document in data:
            title = document["title"]
            full_text = document["text"]
            for annotation in document["annotations"]:
                special_token_text = SpanAnnotation.build_special_token_text_from_json(
                    annotation, full_text, include_special_tokens
                )
                event_type = None
                if annotation.get("prediction") is not None:
                    event_type = EventType.from_tag_name(annotation["predicted"])
                text = full_text[annotation["start"] : annotation["end"]]
                span_anno = SpanAnnotation(
                    text=text,
                    special_token_text=special_token_text,
                    event_type=event_type,
                    iterative=None,
                    mental=None,
                    speech_type=None,
                    thought_representation=None,
                    document_text=full_text,
                    start=annotation["start"],
                    end=annotation["end"],
                    spans=[(s[0], s[1]) for s in annotation["spans"]],
                )
                self.documents[title].append(span_anno)
                self.annotations.append(span_anno)

    def get_annotation_json(
        self, predictions: EvaluationResult
    ) -> List[Dict[str, Any]]:
        out_data = []
        for title, document in self.documents.items():
            out_doc = {"title": title, "text": None, "annotations": []}
            prediction_list = predictions.get_prediction_lists()
            assert len(prediction_list["event_types"]) == len(self.annotations)
            for i, annotation in enumerate(document):
                if out_doc["text"] is None:
                    out_doc["text"] = annotation.document_text
                out_doc["annotations"].append(
                    annotation.output_dict(
                        {
                            k: v[i].to_string() if hasattr(v[i], "to_string") else v[i]
                            for k, v in prediction_list.items()
                        }
                    )
                )
            out_doc["annotations"] = list(
                sorted(out_doc["annotations"], key=lambda d: d["start"])
            )
            out_data.append(out_doc)
        return out_data

    def save_json(self, out_path: str, prediction: EvaluationResult):
        out_data = self.get_annotation_json(prediction)
        out_file = open(out_path, "w")
        json.dump(out_data, out_file)

    def __getitem__(self, i: int):
        return self.annotations[i]

    def __len__(self):
        return len(self.annotations)


def merge_direct_neighbors(selectors):
    to_remove = []
    for i in range(len(selectors) - 1):
        if selectors[i].end == selectors[i + 1].start:
            selectors[i + 1].start = selectors[i].start
            to_remove.append(i)
    return [selector for i, selector in enumerate(selectors) if i not in to_remove]


def merge_direct_neighbors_json(selectors):
    to_remove = []
    for i in range(len(selectors) - 1):
        if selectors[i][1] == selectors[i + 1][0]:
            selectors[i + 1][0] = selectors[i][0]
            to_remove.append(i)
    return [selector for i, selector in enumerate(selectors) if i not in to_remove]
