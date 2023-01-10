# Output Formats

In the default configuration, the script `main.py` outputs its result as tabular data in a pseudo-CONLL format.
Each token is represented on one line, with fields separated by tabs.
The entire output is encoded in plain text (UTF-8, normalized to NFC, using only the LF character as line break,
normalized to NFC in the default configuration).
Unlike the traditional CONNL-U format, sentences are not separated by newlines.
Boolean values are encoded as `0`/`1`.

```
TODO example
```

Fields
* `i`: Index of the token, counted through the document, starting at 0.
* `text`: Text representation / word form of the token.
* `is_sent_start`: takes on value `1` if this token starts a new sentence, cf. [Tokenizer](TODO).
* `is_para_start`: takes on value `1` if this token starts a new paragraph, cf. [Tokenizer](TODO).
* `is_section_start`: takes on value `1` if this token starts a new paragraph, cf. [Tokenizer](TODO).
* `tag`: POS tag of the token from the [TIGER variant of the STTS tagset](https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/tiger-corpus/annotation/tiger_scheme-syntax.pdf) ([Overview](https://www.linguistik.hu-berlin.de/de/institut/professuren/korpuslinguistik/mitarbeiter-innen/hagen/STTS_Tagset_Tiger)), cf. module [tagger_someweta](TODO)
* `pos`: Part-of-Speech tag of the token from the [Universal Dependencies v2 POS tagset](https://universaldependencies.org/u/pos/all.html), automatically converted using table [*de::stts*](https://universaldependencies.org/tagset-conversion/de-stts-uposf.html)
* `lemma`: Lemma of the word form, cf. module [lemma_rnntagger](TODO).
* `head`: Head of the current word, encoded by the head's value `i`. For roots, this is identical to the token's `i`.
* `dep`: Dependency relation of the token's head to this token. Labels are from “Eine umfassende Constraint-Dependenz-Grammatik des Deutschen” ([Forth, 2005](#ref-forth_umfassende_2014)) as labels, used in, e.g., the Hamburg Dependency Treebank ([Overview](https://github.com/rsennrich/ParZu/blob/master/doc/LABELS.md)). Cf. module [parser_parzu](TODO).
* `entity`: NER tag of this token, in [IOB encoding](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)). Uses the usual four classes `PER, LOC, ORG, MISC` as employed by the respective CoNLL-2003 shared task on named entity recognition. ([Sang and De Meulder, 2003](#ref-tjong_kim_sang_introduction_2003)). Cf. module [ner_flair](TODO).
* `speech`: Comma-separated list of speech types annotated on this token, or `_` if none. A subset of `direct`, `indirect`, `freeIndirect` and `reported`, as defined by [Brunner et al. (2021)](#ref-brunner_bert_2021). Cf. module [speech_redewiedergabe](TODO).
* `coref_clusters`: Comma-separated list of coreference clusters (represented by integer IDs) annotated on this token, or `_` if none. Cf. module [coref_uhhlt](TODO).
* `scene_id`: Numerical index of the scene which contains this token, cf. module [scenes_stss_se](TODO).
* `scene_label`: Annotated label of the scene having index `scene_id`. One of `Scene` or `Nonscene`.
* `event_id`: Numerical index of the event which contains this token, or `_` if token is not contained in an event. Cf. module [events_uhhlt](TODO).
* `event_label`: Annotated label of the event having index `event_id`. One of `non_event`, `change_of_state`, `process` or `stative_event`, as defined by [Vauth et al. (2021)](#ref-vauth_automated_2021).

# References

<div id="ref-brunner_bert_2021">

<p>Brunner, Annelen, Ngoc Duyen Tanja Tu, Lukas Weimer, and Fotis Jannidis. 2021. “To BERT or Not to BERT – Comparing Contextual Embeddings in a Deep Learning Architecture for the Automatic Recognition of Four Types of Speech, Thought and Writing Representation.” In <em>Proceedings of the 5th Swiss Text Analytics Conference (SwissText) &amp; 16th Conference on Natural Language Processing (KONVENS)</em>, 2624:11. CEUR Workshop Proceedings. Zurich, Switzerland. <a href="http://ceur-ws.org/Vol-2624/paper5.pdf">http://ceur-ws.org/Vol-2624/paper5.pdf</a>.</p>

</div>

<div id="ref-forth_umfassende_2014" class="csl-entry" role="doc-biblioentry">

Forth, Kilian A. 2014. <em>Eine Umfassende Constraint-Dependenz-Grammatik Des Deutschen</em>. Universität Hamburg. <a href="https://edoc.sub.uni-hamburg.de/informatik/volltexte/2014/204/">https://edoc.sub.uni-hamburg.de/informatik/volltexte/2014/204/</a>.

</div>

<div id="ref-tjong_kim_sang_introduction_2003" class="csl-entry" role="doc-biblioentry">

Sang, Erik F. Tjong Kim, and Fien De Meulder. 2003. <span>“Introduction to the <span>CoNLL</span>-2003 Shared Task: Language-Independent Named Entity Recognition.”</span> In <em>Proceedings of the Seventh Conference on Natural Language Learning at <span>HLT</span>-<span>NAACL</span> 2003</em>, 142–47. Edmonton, Canada: Association for Computational Linguistics. <a href="https://doi.org/10.3115/1119176.1119195">https://doi.org/10.3115/1119176.1119195</a>.

</div>

<div id="ref-vauth_automated_2021">

<p>Vauth, Michael, Hans Ole Hatzel, Evelyn Gius, and Chris Biemann. 2021. “Automated Event Annotation in Literary Texts.” In <em>Proceedings of the Conference on Computational Humanities Research 2021</em>, edited by Maud Ehrmann, Folgert Karsdorp, Melvin Wevers, Tara Lee Andrews, Manuel Burghardt, Mike Kestemont, Enrique Manjavacas, Michael Piotrowski, and Joris van Zundert, 2989:333–45. CEUR Workshop Proceedings. Amsterdam, the Netherlands. <a href="https://ceur-ws.org/Vol-2989/#short_paper18">https://ceur-ws.org/Vol-2989/#short_paper18</a>.</p>

</div>
