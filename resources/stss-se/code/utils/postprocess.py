import itertools
import json
import os

import jsonlines
import sys


def read_jsonlines(file_path):
    content = []
    with jsonlines.open(file_path) as f:
        for line in f.iter():
            content.append(line)
    return content


def post_process2(original_file_path, tmp_file_path, pred_file_path, out_file=None):
    if not out_file:
        out_file = pred_file_path.replace(".pred", "")

    original_file = json.load(open(original_file_path, ))
    pred = read_jsonlines(pred_file_path)
    tmp_file_sent_boundaries = read_jsonlines(tmp_file_path)
    labels = list(itertools.chain(*[line["labels"] for line in pred]))
    labels = [l.replace("_label", "") if "-" in l else "x" for l in labels]
    indexes = [(line[0], line[1]) for line in tmp_file_sent_boundaries[0]["indices"]]
    labels = list(zip(labels, indexes))
    scenes = []
    try:
        for i, l in enumerate(labels):
            if "x" not in l[0]:
                if i == 0:
                    entry = {"begin": l[1][0], "end": -1, "type": l[0].replace("-B", "")}
                else:
                    if entry:
                        entry["end"] = l[1][0]
                        scenes.append(entry)
                        entry = {"begin": l[1][0], "end": -1, "type": l[0].replace("-B", "")}
                        scenes.append(entry)
                    else:
                        entry = {"begin": l[1][0], "end": -1, "type": l[0].replace("-B", "")}

        entry["end"] = l[1][0]
        if entry not in scenes:
            scenes.append(entry)
    except:
        print(original_file_path, "could not!")
        return

    output = {"text": original_file["text"], "scenes": scenes}
    if out_file.endswith("l"):
        out_file = out_file[:-1]
    json.dump(output, open(out_file, "w"))


def post_process(original_file_path, tmp_file_path, pred_file_path, out_file=None):
    if not out_file:
        out_file = pred_file_path.replace(".pred", "")

    original_file = json.load(open(original_file_path, ))
    pred = read_jsonlines(pred_file_path)
    labels = list(itertools.chain(*[line["labels"] for line in pred]))
    tmp_file_sent_boundaries = read_jsonlines(tmp_file_path)

    indexes = [(line[0], line[1]) for line in tmp_file_sent_boundaries[0]["indices"]]
    scenes = []
    labels = list(zip(labels, indexes))
    group = {}
    last_border = 0
    for i, label_offset in enumerate(labels):
        label, offset = label_offset[0].replace("_label", ""), label_offset[1]
        if i == 0:
            prev_l = label.replace("-B", "")
            group = [offset]
        else:
            if "-B" in label:
                # Non-scene to non-scene change is not allowed so continue expanding last non-scene despite
                # prediction of Nonscene-B label
                if label == "Nonscene-B" and prev_l == "Nonscene":
                    group.append(offset)
                else:  # scene change due to prediction of -B label
                    scenes.append({"begin": last_border, "end": group[-1][-1], "type": prev_l})
                    group = [offset]
                    last_border = scenes[-1]["end"]
                    prev_l = label.replace("-B", "")
            else:
                if label == prev_l:
                    group.append(offset)
                else:  # scene change despite lack of -B label
                    scenes.append({"begin": last_border, "end": group[-1][-1], "type": prev_l})
                    group = [offset]
                    last_border = scenes[-1]["end"]
                    prev_l = label.replace("-B", "")
    if group:
        scenes.append({"begin": last_border, "end": group[-1][-1], "type": prev_l})
    output = {"text": original_file["text"], "scenes": scenes}
    if out_file.endswith("l"):
        out_file = out_file[:-1]
    json.dump(output, open(out_file, "w"))
    os.remove(pred_file_path)


if __name__ == "__main__":
    pred_file_path = "33stss_20_30_predictions/9783732522033.json.pred"  # "data/predictions/{}.pred".format(test_file)
    original_file_path = "data/test/9783732522033.json"
    tmp_file = "data/tmp/9783732522033.jsonl"
    post_process(original_file_path, tmp_file, pred_file_path)
