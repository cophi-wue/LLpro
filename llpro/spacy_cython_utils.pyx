# distutils: language=c++
# cython: infer_types=True, bounds_check=False, profile=True

from spacy.tokens.doc cimport Doc
from spacy.tokens.doc cimport set_children_from_heads

def apply_dependency_to_doc(Doc doc, heads, deprels):
    sent_start = [0]*doc.length
    for i in range(doc.length):
        sent_start[i] = doc.c[i].sent_start

    for i in range(doc.length):
        doc.c[i].head = heads[i] - i
        doc.c[i].dep = doc.vocab.strings.add(deprels[i])
    set_children_from_heads(doc.c, 0, doc.length)

    # restitute sentence boundaries
    for i in range(doc.length):
        doc.c[i].sent_start = sent_start[i]
    return doc

