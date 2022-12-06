import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from tqdm import tqdm

from .datasets import EventType, SpeechType
from .evaluation_result import EvaluationResult


def plot_confusion_matrix(
    target,
    hypothesis,
    normalize="true",
    tick_names=["Non Event", "Stative Event", "Process", "Change of State"],
):
    cm = confusion_matrix(target, hypothesis, normalize=normalize)
    cm2 = confusion_matrix(target, hypothesis, normalize=None)
    ax = sns.heatmap(
        cm,
        vmin=0.0,
        vmax=1.0,
        cmap=plt.cm.Blues,
        xticklabels=tick_names,
        yticklabels=tick_names,
        square=True,
        annot=True,
        cbar=True,
    )
    ax.set_ylabel("True Labels")
    ax.set_xlabel("Predicted Labels")
    # print(cm2)
    # disp = ConfusionMatrixDisplay(
    #     confusion_matrix=cm,
    #     display_labels=["non-event", "change of state", "process", "stative event"],
    # )
    # plotted = disp.plot(
    #     cmap=plt.cm.Blues,
    # )
    # plotted.figure.colorbar()
    return ax


@torch.no_grad()
def evaluate(
    loader, model, device=None, out_file=None, save_confusion_matrix=False, epoch=None
):
    model.to(device)
    if device is None:
        device = model.device
    model.eval()
    gold = []
    predictions = []
    texts = defaultdict(list)
    labled_annotations = []
    all_predictions = defaultdict(list)
    all_labels = defaultdict(list)
    for input_data, gold_labels, annotations in tqdm(loader, desc="Evaluating"):
        out = model(**input_data.to(device))
        for anno, label in zip(annotations, out.event_type.cpu()):
            labled_annotations.append((label, anno))
        if out_file is not None:
            not_non_event_index = 0  # Counted up whenever an event is not a non-event
            for i, anno in enumerate(annotations):
                out_data = anno.output_dict(
                    {"event_types": EventType(out.event_type[i].item()).to_string()}
                )
                out_data["gold_label"] = EventType(
                    gold_labels.event_type[i].item()
                ).to_string()
                out_data["additional_predictions"] = dict()
                out_data["additional_labels"] = dict()
                if anno.event_type != EventType.NON_EVENT:
                    for prop in ["mental", "iterative"]:
                        out_data["additional_labels"][prop] = bool(
                            getattr(gold_labels, prop)[not_non_event_index].item()
                        )
                        out_data["additional_predictions"][prop] = bool(
                            getattr(out, prop)[not_non_event_index].item()
                        )
                out_data["additional_labels"]["speech_type"] = SpeechType(
                    gold_labels.speech_type[i].item()
                ).to_string()
                out_data["additional_predictions"]["speech_type"] = SpeechType(
                    out.speech_type[i].item()
                ).to_string()
                out_data["additional_labels"]["thought_representation"] = bool(
                    gold_labels.thought_representation[i].item()
                )
                out_data["additional_predictions"]["thought_representation"] = bool(
                    out.thought_representation[i].item()
                )
                texts[anno.document_text].append(out_data)
                if anno.event_type != EventType.NON_EVENT:
                    not_non_event_index += 1
        predictions.append(out.event_type.cpu())
        for name in ["mental", "iterative"]:
            if gold_labels is not None:
                selector = gold_labels.event_type != 0
            else:
                selector = out.event_type != 0
            all_predictions[name].append(
                torch.masked_select(getattr(out, name).cpu(), selector.cpu())
            )
            if gold_labels is not None:
                all_labels[name].append(getattr(gold_labels, name).cpu())
                assert len(all_labels[name][-1]) == len(all_predictions[name][-1])
        for name in ["speech_type", "thought_representation"]:
            all_predictions[name].append(getattr(out, name).cpu())
            if gold_labels is not None:
                all_labels[name].append(getattr(gold_labels, name).cpu())
        if gold_labels is not None:
            gold.append(gold_labels.event_type.cpu())
    for label, annotation in labled_annotations:
        logging.debug(
            f"=== Gold: {annotation.event_type}, predicted: {EventType(label.item())}"
        )
        logging.debug(annotation.text)
    if len(gold) == 0:
        logging.warning("No gold labels given, not calculating classification report")
        report = None
    else:
        report = classification_report(
            torch.cat(gold), torch.cat(predictions), output_dict=True
        )
        logging.info(classification_report(torch.cat(gold), torch.cat(predictions)))
    if (save_confusion_matrix or epoch is not None) and len(gold) > 0:
        plt.rc("text")
        plt.rc("font", family="serif", size=12)
        gold_data_converted = torch.tensor(
            [
                EventType(gold_label.item()).get_narrativity_ordinal()
                for gold_label in torch.cat(gold)
            ],
            dtype=torch.int,
        )
        predictions_converted = torch.tensor(
            [
                EventType(prediction.item()).get_narrativity_ordinal()
                for prediction in torch.cat(predictions)
            ],
            dtype=torch.int,
        )
        _ = plot_confusion_matrix(gold_data_converted, predictions_converted)
        plt.tight_layout()
        plt.gcf().subplots_adjust(left=0.2)
        plt.savefig(f"confusion_matrix_event-types_{epoch}.pdf")
        mlflow.log_artifact(f"confusion_matrix_event-types_{epoch}.pdf")
        plt.clf()
    if out_file is not None:
        out_list = []
        for text, annotations in texts.items():
            out_list.append(
                {
                    "text": text,
                    "title": None,
                    "annotations": annotations,
                }
            )
        json.dump(out_list, out_file)
    all_predictions = {
        name: torch.cat(values) for name, values in all_predictions.items()
    }
    all_labels = {name: torch.cat(values) for name, values in all_labels.items()}
    if report is not None:
        extra_metrics = {}
        event_kind_report = None
        for name, values in all_predictions.items():
            plt.rc("text")
            plt.rc("font", family="serif", size=12)
            plot_confusion_matrix(all_labels[name], values, tick_names="auto")
            report = classification_report(all_labels[name], values, output_dict=True)
            if name == "speech_type":
                print("Number of character speech examples:", report["0"]["support"])
                print("Number of narrator speech examples:", report["1"]["support"])
                print(
                    classification_report(
                        all_labels[name],
                        values,
                    )
                )
            if name == "edge_case_speech_type":
                print(
                    "Number of character speech examples with filtering:",
                    report["0"]["support"],
                )
                print(
                    "Number of narrator speech examples with filtering:",
                    report["1"]["support"],
                )
                print(
                    classification_report(
                        all_labels[name],
                        values,
                    )
                )
            if name == "speech_type":
                event_kind_report = report
            extra_metrics[name + " macro f1"] = report["macro avg"]["f1-score"]
            extra_metrics[name + " weighted f1"] = report["weighted avg"]["f1-score"]
            try:
                extra_metrics[name + " _ 1"] = report["1.0"]["f1-score"]
                extra_metrics[name + " _ 0"] = report["0.0"]["f1-score"]
            except KeyError:
                extra_metrics[name + " _ 1"] = report["1"]["f1-score"]
                extra_metrics[name + " _ 0"] = report["0"]["f1-score"]
            plt.tight_layout()
            plt.gcf().subplots_adjust(left=0.2)
            file_name = (
                f"confusion_matrix_{name}_{epoch if epoch is not None else 'end'}.pdf"
            )
            plt.savefig(file_name)
            plt.clf()
            mlflow.log_artifact(file_name)
        return EvaluationResult(
            weighted_f1=event_kind_report["weighted avg"]["f1-score"],
            macro_f1=event_kind_report["macro avg"]["f1-score"],
            predictions=torch.cat(predictions).cpu(),
            extra_metrics=extra_metrics,
            extra_labels=all_labels,
            extra_predictions=all_predictions,
        )
    else:
        return EvaluationResult(
            weighted_f1=None,
            macro_f1=None,
            predictions=torch.cat(predictions).cpu()
            if len(predictions) > 0
            else torch.tensor([]),
            extra_labels=all_labels,
            extra_predictions=all_predictions,
        )
