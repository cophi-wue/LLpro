"""
Change clipboard content from script output to pasteable row for archival purposes.

1. Copy output starting from '====== TOTALS =======' up to and including 'Official avg F1: [...]'
2. Run this script
3. Paste row into sheets/excel/calc (format: recall, precision,f1 for muc, bcub and ceafe followed by avg f1)
"""
import pyperclip
import re

RECALL_PATTERN = re.compile("Recall:\s\(.*?\)\s(\d+\.\d+)%")
PRECISION_PATTERN = re.compile("Precision:\s\(.*?\)\s(\d+\.\d+)%")
F1_PATTERN = re.compile("F1:\s(\d+\.?\d+)%")
FINAL_F1_PATTERN = re.compile("Official avg F1:\s(\d+\.\d+)$")

text = pyperclip.paste()


recall = RECALL_PATTERN.findall(text)
precisions = PRECISION_PATTERN.findall(text)
f1 = F1_PATTERN.findall(text)
final_f1 = FINAL_F1_PATTERN.findall(text)
assert len(recall) == 6
assert len(precisions) == 6
assert len(f1) == 6
assert len(final_f1) == 1

assert recall[0] == recall[2]
assert recall[2] == recall[4]
assert precisions[0] == precisions[2]
assert precisions[2] == precisions[4]
assert f1[0] == f1[2]
assert f1[2] == f1[4]

new_text = (
    f"{recall[1]}%\t{precisions[1]}%\t{f1[1]}%\t{recall[3]}%\t{precisions[3]}%\t{f1[3]}%\t{recall[5]}%\t{precisions[5]}%\t{f1[5]}%\t{final_f1[0]}%"
)
pyperclip.copy(new_text)
