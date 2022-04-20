#!/usr/bin/env python3
"""
Perform split to produce same test/dev set as tuba10 with slightly larger training set.
"""
import sys

f = open(sys.argv[1])

splits = [(727, "test.german.tuba11_gold_conll"), (727, "dev.german.tuba11_gold_conll"), (2362, "train.german.tuba11_gold_conll")]

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
