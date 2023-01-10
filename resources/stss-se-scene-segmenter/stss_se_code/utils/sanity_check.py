import glob
import itertools
import logging
from pathlib import Path

import jsonlines
import numpy as np
from sklearn.metrics import f1_score

from eval import eval_one_file

logging.getLogger().setLevel(logging.DEBUG)


def read_jsonlines(file_path):
    content = []
    with jsonlines.open(file_path) as f:
        for line in f.iter():
            content.append(line)
    return content


def eval_folder(gold_dir: Path, pred_dir: Path):

    for gold_file in gold_dir.iterdir():
        pred_file = pred_dir.joinpath(gold_file.name.replace("jsonl", "json.pred"))
        if eval_one_file and eval_one_file not in gold_file.name:
            continue
        if not pred_file.is_file():
            print(
                "Missing annotations for file %s! Please write all predictions to the folder `/predictions` with the same "
                "name as the input file" % str(
                    gold_file))
            continue
        original_file = read_jsonlines(gold_file)
        pred = list(itertools.chain(*[line["labels"] for line in read_jsonlines(pred_file)]))
        pred = [p.replace("_label", "") for p in pred]
        labels = sorted(set(original_file[0]["labels"][:len(pred)]))
        b_labels = [i for i, l in enumerate(labels) if "-B" in l ]
        print(gold_file, labels)
        print(f1_score(original_file[0]["labels"][:len(pred)], pred, average=None, labels=labels))
        print(f1_score(original_file[0]["labels"][:len(pred)], pred, average="macro", labels=labels))
        print(np.average(np.array(f1_score(original_file[0]["labels"][:len(pred)], pred, average=None, labels=labels))[b_labels]))

        print()

if __name__ == '__main__':
    gold_dir = Path("../../data/tmp")
    listing = glob.glob('*predictions*/')
    for p in listing:
        print(p)
        pred_dir = Path(p)
        eval_folder(gold_dir=gold_dir, pred_dir=pred_dir)
        print()
