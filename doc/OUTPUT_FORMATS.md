# Output Formats

# Tagsets

* POS Tags from SoMeWeTa: uses [TIGER variant of the STTS tagset](https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/tiger-corpus/annotation/tiger_scheme-syntax.pdf) (TODO) (field `someweta_pos_stts`). [Overview](TODO). Additonally, provides automatic conversion to [Universal Dependencies v2 POS tags](https://universaldependencies.org/u/pos/all.html) including [Features](https://universaldependencies.org/u/feat/all.html). (fields `someweta_pos_ud_tag`, `someweta_pos_ud_feats`) Conversion performed using table [*de::stts*](https://universaldependencies.org/tagset-conversion/de-stts-uposf.html).
* Dependency relation labels from Parzu: uses “Eine umfassende Constraint-Dependenz-Grammatik des Deutschen” (Forth, 2005) as used in, e.g., the Hamburg Dependency Treebank. [Overview](https://github.com/rsennrich/ParZu/blob/master/doc/LABELS.md).
* Morphological Analysis from RNNTagger: uses the [TIGER Morphology Annotation Scheme](https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/tiger-corpus/annotation/tiger_scheme-morph.pdf).
* Named Entites from FLERT: `PER, LOC, ORG, MISC` as employed by the spective CoNLL-2003 shared task on named entity recognition. (TODO)
* Semantic Role Labels from InVeRo-XL: Frames and Semantic Roles defined by VerbAtlas ([Di Fabio et al.](#)). [List of Semantic Roles](https://verbatlas.org/semantic), [List of Frames](https://verbatlas.org/frames).
