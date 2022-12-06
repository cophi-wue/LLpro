"""
Preprocess raw documents to perform event classification on.
"""
import bisect
import json
import os
import re
from typing import List, Optional

import catma_gitlab as catma
import spacy
import typer

from event_classify.datasets import SimpleEventDataset
from event_classify.parser import HermaParser, Parser, ParZuParser
from event_classify.preprocessing import build_pipeline, get_annotation_dicts, use_gpu

app = typer.Typer()


@app.command()
def preprocess(
    text_file_paths: List[str],
    out_file_path: str,
    title: Optional[str] = None,
    gpu: bool = False,
    parser: Parser = typer.Option(Parser.SPACY),
):
    """
    Segment a set document into event spans based on verb occurrences.

    Creates a JSON file with the document and its event spans suitable for passing to the `predict.py`.
    """
    if gpu:
        use_gpu()

    nlp = build_pipeline(parser)
    document_list = []
    for text_file_path in text_file_paths:
        print(f"Processing {text_file_path}")
        in_file = open(text_file_path, "r")
        full_text = "".join(in_file.readlines())
        doc = nlp(full_text)
        inferred_title, _ = os.path.splitext(os.path.basename(text_file_path))
        data = {"text": full_text, "title": title or inferred_title, "annotations": []}
        data["annotations"] = get_annotation_dicts(doc)
        document_list.append(data)
    json.dump(document_list, open(out_file_path, "w"))


@app.command()
def spans(
    input_sentence: str,
    display: bool = False,
    parser: Parser = typer.Option(Parser.SPACY),
):
    nlp = build_pipeline(parser)
    doc = nlp(input_sentence)
    if display:
        spacy.displacy.serve(doc, style="dep")
    for token in doc:
        print(token, token.tag_, token.pos_, token.dep_, token._.custom_dep)
    for ranges in doc._.events:
        print("======")
        for event_range in ranges:
            print(event_range)


def is_close(predict, target, limit=5):
    return abs(predict[0] - target[0]) <= limit and abs(predict[1] - target[1]) <= limit


@app.command()
def eval(parser: Parser = typer.Option(Parser.SPACY)):
    """
    Evaluate segmentation outputs.
    """
    # set_gpu_allocator("pytorch")
    # require_gpu(0)
    gold_external_spans = set()
    verwandlung_dataset, text = get_verwandlung()
    texts = []
    text_spans = []
    for annotation in verwandlung_dataset:
        for span in annotation.spans:
            new_text = re.subn("[^A-Za-z]", "", text[span[0] : span[1]])[0]
            texts.append(new_text)
            text_spans.append(span)
        # gold_external_spans.add((annotation.start, annotation.end))
    nlp = build_pipeline(parser)
    doc = nlp(text)
    predict_external_spans = set()
    tp = 0
    fp = 0
    matched = [False for _ in range(len(texts))]
    for event_ranges in doc._.events:
        min_start = min(r.start_char for r in event_ranges)
        max_end = max(r.end_char for r in event_ranges)
        to_print = ""
        # to_print = to_print + text[min_start - 100:min_start]
        previous_end = min_start - 100
        for r in event_ranges:
            to_print = (
                to_print
                + text[previous_end : r.start_char]
                + "<EVENT>"
                + text[r.start_char : r.end_char]
                + "</EVENT>"
            )
            previous_end = r.end_char
        to_print = to_print + text[max_end : max_end + 100]
        print("=======")
        if len(event_ranges) > 1:
            print(to_print)
        for r in event_ranges:
            cleaned_text = re.subn("[^A-Za-z]", "", r.text)[0]
            match_ids = [i for i, t in enumerate(texts) if cleaned_text == t]
            match_spans = [text_spans[i] for i in match_ids]
            matches = [is_close((r.start_char, r.end_char), ms) for ms in match_spans]
            matched_to = [
                i
                for i in match_ids
                if is_close((r.start_char, r.end_char), text_spans[i])
            ]
            if any(matches):
                matched[matched_to[0]] = True
                tp += 1
            else:
                fp += 1
    precision = tp / (tp + fp)
    fn = len([e for e in matched if e is False])
    recall = tp / (tp + fn)
    print(precision, recall)
    print("F1:", 2 * (precision * recall) / (precision + recall))


def find_best_match_for_start(sorted_list, element):
    i = bisect.bisect_left(sorted_list, element)
    if i != len(sorted_list):
        return i
    raise ValueError


def get_verwandlung():
    project = catma.CatmaProject(
        ".",
        "CATMA_DD5E9DF1-0F5C-4FBD-B333-D507976CA3C7_EvENT_root",
        filter_intrinsic_markup=False,
    )
    collection = project.ac_dict["Verwandlung_MV"]
    dataset = SimpleEventDataset(
        project,
        ["Verwandlung_MV"],
    )
    return dataset, collection.text.plain_text


if __name__ == "__main__":
    app()
