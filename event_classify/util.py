import bisect
import os
from typing import List, NamedTuple, Optional

from omegaconf import OmegaConf
from transformers import ElectraTokenizer


def get_config(hydra_path):
    conf = OmegaConf.load(os.path.join(hydra_path, "config.yaml"))
    # conf_overrides = OmegaConf.load(os.path.join(hydra_path, "overrides.yaml"))
    overrides_conf = OmegaConf.load(os.path.join(hydra_path, "overrides.yaml"))
    conf.merge_with_dotlist(list(overrides_conf))
    return conf


def get_model(model_path: str, config: Optional[OmegaConf] = None):
    from .datasets import EventType
    from .model import ElectraForEventClassification

    if config is None:
        event_config = get_config(os.path.join(model_path, ".hydra"))
    else:
        event_config = config
    tokenizer: ElectraTokenizer = ElectraTokenizer.from_pretrained(
        os.path.join(model_path, "tokenizer")
    )
    model = ElectraForEventClassification.from_pretrained(
        os.path.join(model_path, "best-model"),
        num_labels=4,
        event_config=event_config,
    )
    return model, tokenizer


def filter_sorted(
    data, start_value, end_value, key=lambda x: x, key_start=None, key_end=None
):
    if key_start is None:
        key_start = key
    if key_end is None:
        key_end = key
    start_index = bisect.bisect_left([key_start(el) for el in data], start_value)
    end_index = bisect.bisect_right([key_end(el) for el in data], end_value)
    return data[start_index:end_index]


def smooth_bins(annotation_spans: List, smooth_character_span=1000) -> List[int]:
    """
    Smooth narrativity score for all event in a text segment.

    This function is based on Michael Vauth's code.
    """
    smooth_narrativity_scores = []
    start_points = []

    max_end = max(span["end"] for span in annotation_spans)
    starts = [span["start"] for span in annotation_spans]
    ends = [span["end"] for span in annotation_spans]
    for b in range(
        int(smooth_character_span / 2), max_end, 100
    ):  # iterates in 100 character steps over the text
        start_points.append(b)
        start_index = bisect.bisect_left(starts, b - smooth_character_span)
        end_index = bisect.bisect_right(ends, b + smooth_character_span)
        annotation_spans_filtered = annotation_spans[start_index : end_index - 10]
        # The ends are not necessarily sorted (sorted by start), so let's check around here if there is anything we need to fix
        for anno_span in annotation_spans[end_index - 10 : end_index + 10]:
            if (anno_span["start"] > b - smooth_character_span) and (
                anno_span["end"] < b + smooth_character_span
            ):
                annotation_spans_filtered.append(anno_span)

        smooth_narrativity_scores.append(
            sum(
                EventType.from_tag_name(span["predicted"]).get_narrativity_score()
                for span in annotation_spans_filtered
            )
        )
    return smooth_narrativity_scores


class SubDoc(NamedTuple):
    offset: int
    text: str


def split_text(text: str, allowed_split=".\n") -> List[SubDoc]:
    """
    Split text into a number of sub strings managable for spacy.

    This could fail for abritrary sequences but the books in d-prose all have
    seem to have '.\n'
    """
    total = len(text)
    # This is super conservative spacy seems to be able to do >300k tokens on 12GB VRAM
    max_segment_length = 100000
    segments = text.split(allowed_split)
    out = []
    current_split = []
    for i, segment in enumerate(segments):
        if len(segment) > max_segment_length:
            if allowed_split != ". ":
                # Try again with the double newline strategy
                return split_text(text, allowed_split=". ")
            else:
                raise ValueError("Document has too few split options.")
        if (sum(len(s) for s in current_split) + len(segment)) <= max_segment_length:
            if i == len(segments) - 1:
                current_split.append(segment)
            else:
                current_split.append(segment + allowed_split)
        else:
            out.append(
                SubDoc(
                    text="".join(current_split),
                    offset=sum(len(split.text) for split in out),
                )
            )
            if i == len(segments) - 1:
                current_split = [segment]
            else:
                current_split = [segment + allowed_split]
    out.append(
        SubDoc(
            text="".join(current_split), offset=sum(len(split.text) for split in out)
        )
    )
    return out
