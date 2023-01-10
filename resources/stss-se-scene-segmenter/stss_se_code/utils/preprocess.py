import os
import json
import shutil
from collections import Counter

test_file = "9783732522033.json"
split_dict = {"9783740941093.json": "dev.jsonl", test_file: "test.jsonl"}
all_labels = []


def read_json(json_file, out_dir, use_filename_as_split=False):
    content = json.load(open(json_file, ))
    sentences, labels, indices = [], [], []
    selected = {"Scene": [], "Nonscene": []}
    scene_borders = {range(k["begin"], k["end"]): k["type"] for k in content["scenes"]}
    for sent in content["sentences"]:
        label = None
        for k, v in scene_borders.items():
            if sent["begin"] in k:
                if k not in selected[v]:
                    label = "{}-B".format(v)
                    selected[v].append(k)
                else:
                    label = v
                break
        if not label:
            continue
        sentences.append(content["text"][sent["begin"]:sent["end"]])
        indices.append((sent["begin"], sent["end"]))
        labels.append(label)
    assert len(sentences) == len(labels) == len(indices)
    all_labels.extend(labels)
    print(Counter(labels), (Counter(labels)["Scene-B"] + Counter(labels)["Nonscene-B"]))
    if not use_filename_as_split:
        split = split_dict.get(json_file.split("/")[-1], "train.jsonl")
    else:
        split = json_file.split("/")[-1] + "l"

    with open(os.path.join(out_dir, split), 'a+', encoding="utf8") as outfile:
        batch = {"sentences": sentences, "indices": indices, "labels": labels, "file": json_file.split("/")[-1]}
        json.dump(batch, outfile)
        outfile.write('\n')


if __name__ == "__main__":

    raw_data = "/home/murathan/Desktop/scene-segmentation/json" if "home/" in os.getcwd() else "/cephyr/users/murathan/Alvis/scene-segmentation/json"
    out_dir = "../data/ss"
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    raw_books = sorted([os.path.join(raw_data, l) for l in os.listdir(raw_data)])
    for book in raw_books:
        print(book)
        read_json(book, out_dir)
    total = sum(Counter(all_labels).values())
    print([(k, total / v) for k, v in Counter(all_labels).items()])
