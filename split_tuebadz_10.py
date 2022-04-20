#!/usr/bin/env python3
"""
Perfrom 20-20-60 split as performed in IMS HotCoref paper.
"""
import sys

f = open(sys.argv[1])

splits = [(727, "test.german.tuba10_gold_conll"), (727, "dev.german.tuba10_gold_conll"), (2190, "train.german.tuba10_gold_conll")]

out_files = []
for _, split in splits:
    out_files.append(open(split, "w"))


def get_current_file(out_files, splits, doc_idx):
    total = 0
    for i, (required, _) in enumerate(splits):
        total += required
        if total > doc_idx:
            return out_files[i]


doc_idx = 0
for line in f:
    out = get_current_file(out_files, splits, doc_idx)
    out.write(line)
    if line.strip() == "#end document":
        doc_idx += 1
