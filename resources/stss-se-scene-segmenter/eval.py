import json
from collections import deque, defaultdict
import logging
from logging import info
from pathlib import Path
from pprint import pprint, pformat
from typing import Dict

import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix

logging.getLogger().setLevel(logging.DEBUG)


def eval_file(gold_path: Path, pred_path: Path) -> Dict:
    logging.debug("Comparing files %s and %s..." % (str(gold_path), str(pred_path)))

    data = {}
    with open(str(gold_path)) as f:
        data["gold"] = json.load(f)
    with open(str(pred_path)) as f:
        data["pred"] = json.load(f)
    assert data["gold"]["text"] == data["pred"]["text"], "Mismatch in text between gold and pred text!"
    scenes, boundaries = {}, {}
    for annotation_type in ("gold", "pred"):
        scenes[annotation_type] = data[annotation_type]["scenes"]
        boundaries[annotation_type] = deque()
        prev_typ = scenes[annotation_type][0]["type"]

        for i, scene in enumerate(scenes[annotation_type][1:]):
            begin = scene["begin"]
            typ = scene["type"]

            boundaries[annotation_type].append((begin, "%s-to-%s" % (prev_typ, typ)))

            prev_typ = typ

    logging.debug(pformat(boundaries))

    label_to_int = defaultdict(lambda: len(label_to_int))
    label_to_int["NOBORDER"] = 0

    max_index = len(data["gold"]["text"])
    gold_labels = np.zeros(max_index)
    pred_labels = np.zeros(max_index)

    for boundary in boundaries["pred"]:
        pred_labels[boundary[0]] = label_to_int[boundary[1]]
    for boundary in boundaries["gold"]:
        gold_labels[boundary[0]] = label_to_int[boundary[1]]
    int_to_labels = {value: key for key, value in label_to_int.items()}

    info(classification_report(y_true=gold_labels, y_pred=pred_labels,
                               target_names=[int_to_labels[i] for i in range(1, len(int_to_labels))],
                               labels=[1, 2, 3]))
    info(confusion_matrix(y_true=gold_labels, y_pred=pred_labels, labels=[1, 2, 3]))

    return classification_report(y_true=gold_labels, y_pred=pred_labels,
                                 target_names=[int_to_labels[i] for i in range(1, len(int_to_labels))],
                                 labels=[1, 2, 3], output_dict=True)


def eval_folder(gold_dir: Path, pred_dir: Path):
    results = []
    f1_scores = []

    for gold_file in gold_dir.iterdir():
        pred_file = pred_dir.joinpath(gold_file.name)
        if not pred_file.is_file():
            raise RuntimeError(
                "Missing annotations for file %s! Please write all predictions to the folder `/predictions` with the same "
                "name as the input file" % str(
                    gold_file))
        result = eval_file(gold_path=gold_file, pred_path=pred_file)
        results.append(result)
        f1_scores.append(result['macro avg']["f1-score"])

    logging.info("Mean macro avg. f1 score over all files: %.2f" % np.mean(f1_scores))


if __name__ == '__main__':
    gold_dir = Path("/data/test")
    pred_dir = Path("/predictions")

    eval_folder(gold_dir=gold_dir, pred_dir=pred_dir)

    print()