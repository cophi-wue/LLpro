import bisect
import json
import sys

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report

from event_classify.datasets import EventType
from event_classify.eval import plot_confusion_matrix

data_gold = json.load(open(sys.argv[1], "r"))
data_pipeline = json.load(open(sys.argv[2], "r"))

matches = []

annos_1 = []

num_matches = 0
correct_matches = 0

first_file_strategy =  "gold_label" #"predicted" # or
for annotation in data_gold[0]["annotations"]:
    annos_1.append(annotation["start"])

for annotation in data_pipeline[0]["annotations"]:
    index, _ = min(enumerate(annos_1), key=lambda x: abs(x[1] - annotation["start"]))
    matched = data_gold[0]["annotations"][index]
    # if matched["spans"] == annotation["spans"]:
    num_matches += 1
    gold_start, gold_end = matched["start"], matched["end"]
    start, end = annotation["start"], annotation["end"]
    print(data_gold[0]["text"][start:end])
    print(data_gold[0]["text"][gold_start:gold_end])
    if matched["start"] == annotation["start"]:
        correct_matches += 1
    print(gold_start, start)
    print("Labels:", matched[first_file_strategy], annotation["predicted"])
    matches.append((matched[first_file_strategy], annotation["predicted"]))

trues = []
preds = []
for first, second in matches:
    trues.append(EventType.from_tag_name(first).value)
    preds.append(EventType.from_tag_name(second).value)
print("Match start accuracy: ", correct_matches / num_matches)
print(classification_report(trues, preds))
gold_data_converted = torch.tensor([EventType(label).get_narrativity_ordinal() for label in trues], dtype=torch.int)
predictions_converted = torch.tensor([EventType(prediction).get_narrativity_ordinal() for prediction in preds], dtype=torch.int)
_ = plot_confusion_matrix(gold_data_converted, predictions_converted)
plt.show()
