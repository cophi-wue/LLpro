# Output Formats

In the default configuration, the script `main.py` outputs its result as tabular data in a pseudo-CONLL format.
Each token is represented on one line, with fields separated by tabs.
The entire output is encoded in plain text (UTF-8, normalized to NFC, using only the LF character as line break,
normalized to NFC in the default configuration).
Unlike the traditional CONNL-U format, sentences are not separated by newlines.
Boolean values are encoded as `0`/`1`.

Example (where, for readability, tabs are replaced with spaces to align the columns):
```
i   text                is_sent_start  is_para_start  is_section_start  pos    tag      lemma             morph                                                       dep   head  speech  entity  coref_clusters  scene_id  scene_label  event_id  event_label    
0   Aus                 1              1              _                 ADP    APPR     aus               AdpType=Prep                                                pp    10    _       O       _               0         Scene        _         _              
1   dem                 0              0              _                 DET    ART      der               Case=Dat|Gender=Neut|Number=Sing|PronType=Art               det   3     _       O       _               0         Scene        _         _              
2   seligsten           0              0              _                 ADJ    ADJA     selig             Case=Dat|Degree=Sup|Gender=Neut|Number=Sing                 attr  3     _       O       0               0         Scene        _         _              
3   Lachen              0              0              _                 NOUN   NN       lachen            Case=Dat|Gender=Neut|Number=Sing                            pn    0     _       O       0               0         Scene        _         _              
4   in                  0              0              _                 ADP    APPR     in                AdpType=Prep                                                pp    3     _       O       _               0         Scene        _         _              
5   das                 0              0              _                 DET    ART      der               Case=Acc|Gender=Neut|Number=Sing|PronType=Art               det   7     _       O       _               0         Scene        _         _              
6   lauteste            0              0              _                 ADJ    ADJA     laut              Case=Acc|Degree=Sup|Gender=Neut|Number=Sing                 attr  7     _       O       _               0         Scene        _         _              
7   Schluchzen          0              0              _                 NOUN   NN       schluchzen        Case=Acc|Gender=Neut|Number=Sing                            pn    4     _       O       _               0         Scene        _         _              
8   der                 0              0              _                 DET    ART      der               Case=Gen|Gender=Fem|Number=Sing|PronType=Art                det   9     _       O       _               0         Scene        _         _              
9   Rührung             0              0              _                 NOUN   NN       Rührung           Case=Gen|Gender=Fem|Number=Sing                             gmod  7     _       O       _               0         Scene        _         _              
10  überspringend       0              0              _                 ADJ    ADJD     überspringend     Degree=Pos|Variant=Short                                    root  0     _       O       _               0         Scene        _         _              
11  ,                   0              0              _                 PUNCT  $,       ,                 PunctType=Comm                                              root  0     _       O       _               0         Scene        _         _              
12  streifte            0              0              _                 VERB   VVFIN    streifen          Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin       root  0     _       O       _               0         Scene        0         process        
13  er                  0              0              _                 PRON   PPER     er                Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs      subj  12    _       O       1               0         Scene        0         process        
14  mehr                0              0              _                 DET    PIAT     mehr              Case=*|Gender=*|Number=*|PronType=Ind,Neg,Tot               obja  12    _       O       _               0         Scene        0         process        
15  als                 0              0              _                 CCONJ  KOKOM    als               ConjType=Comp                                               kom   12    _       O       _               0         Scene        0         process        
16  einmal              0              0              _                 ADV    ADV      einmal            _                                                           cj    15    _       O       _               0         Scene        0         process        
17  ans                 0              0              _                 ADP    APPRART  an                AdpType=Prep|Case=Acc|Gender=Neut|Number=Sing|PronType=Art  pp    12    _       O       _               0         Scene        0         process        
18  Lächerliche         0              0              _                 NOUN   NN       lächerliche       Case=Acc|Gender=Neut|Number=Sing                            pn    17    _       O       _               0         Scene        0         process        
19  und                 0              0              _                 CCONJ  KON      und               _                                                           kon   12    _       O       _               0         Scene        1         process        
20  machte              0              0              _                 VERB   VVFIN    machen            Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin       cj    19    _       O       _               0         Scene        1         process        
21  endlich             0              0              _                 ADV    ADV      endlich           _                                                           adv   20    _       O       _               0         Scene        1         process        
22  nach                0              0              _                 ADP    APPR     nach              AdpType=Prep                                                pp    20    _       O       _               0         Scene        1         process        
23  vollzogener         0              0              _                 ADJ    ADJA     vollzogen         Case=Dat|Degree=Pos|Gender=Fem|Number=Sing                  attr  24    _       O       _               0         Scene        1         process        
24  Feierlichkeit       0              0              _                 NOUN   NN       Feierlichkeit     Case=Dat|Gender=Fem|Number=Sing                             pn    22    _       O       _               0         Scene        1         process        
25  seinem              0              0              _                 DET    PPOSAT   sein              Case=Dat|Gender=Neut|Number=Sing|Poss=Yes|PronType=Prs      det   26    _       O       1               0         Scene        1         process        
26  Entzücken           0              0              _                 NOUN   NN       Entzücken         Case=Dat|Gender=Neut|Number=Sing                            objd  20    _       O       _               0         Scene        1         process        
27  mit                 0              0              _                 ADP    APPR     mit               AdpType=Prep                                                pp    20    _       O       _               0         Scene        1         process        
28  dem                 0              0              _                 DET    ART      der               Case=Dat|Gender=Masc|Number=Sing|PronType=Art               det   30    _       O       _               0         Scene        1         process        
29  wiederholten        0              0              _                 ADJ    ADJA     wiederholt        Case=Dat|Degree=Pos|Gender=Masc|Number=Sing                 attr  30    _       O       _               0         Scene        1         process        
30  Ausrufe             0              0              _                 NOUN   NN       Ausruf            Case=Dat|Gender=Masc|Number=Sing                            pn    27    _       O       _               0         Scene        1         process        
31  :                   0              0              _                 PUNCT  $.       :                 PunctType=Peri                                              root  0     _       O       _               0         Scene        _         _              
32  »                   0              0              _                 PUNCT  $(       <unk>             PunctType=Brck                                              root  0     direct  O       _               0         Scene        _         _              
33  Sie                 0              0              _                 PRON   PPER     sie               Case=Nom|Gender=Fem|Number=Sing|Person=3|PronType=Prs       subj  34    direct  O       1               0         Scene        2         stative_event  
34  ist                 0              0              _                 AUX    VAFIN    sein              Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin       root  0     direct  O       _               0         Scene        2         stative_event  
35  versorgt            0              0              _                 VERB   VVPP     versorgen         Aspect=Perf|VerbForm=Part                                   aux   34    direct  O       _               0         Scene        2         stative_event  
36  ,                   0              0              _                 PUNCT  $,       ,                 PunctType=Comm                                              root  0     direct  O       _               0         Scene        _         _              
37  sie                 0              0              _                 PRON   PPER     sie               Case=Nom|Gender=Fem|Number=Sing|Person=3|PronType=Prs       subj  38    direct  O       1               0         Scene        2         stative_event  
38  ist                 0              0              _                 AUX    VAFIN    sein              Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin       kon   34    direct  O       _               0         Scene        2         stative_event  
39  versorgt            0              0              _                 VERB   VVPP     versorgen         Aspect=Perf|VerbForm=Part                                   aux   34    direct  O       _               0         Scene        2         stative_event  
40  !                   0              0              _                 PUNCT  $.       <unk>             PunctType=Peri                                              root  0     direct  O       _               0         Scene        _         _              
41  «                   0              0              _                 PUNCT  $(       <unk>             PunctType=Brck                                              root  0     direct  O       _               0         Scene        _         _              
42  in                  0              0              _                 ADP    APPR     in                AdpType=Prep                                                root  0     _       O       _               0         Scene        _         _              
43  verschwenderischen  0              0              _                 ADJ    ADJA     verschwenderisch  Case=Dat|Degree=Pos|Gender=Fem|Number=Plur                  attr  44    _       O       _               0         Scene        _         _              
44  Küssen              0              0              _                 NOUN   NN       Kuß               Case=Dat|Gender=Fem|Number=Plur                             pn    42    _       O       _               0         Scene        _         _              
45  und                 0              0              _                 CCONJ  KON      und               _                                                           kon   44    _       O       _               0         Scene        _         _              
46  Umarmungen          0              0              _                 NOUN   NN       Umarmung          Case=Dat|Gender=Fem|Number=Plur                             cj    45    _       O       _               0         Scene        _         _              
47  Luft                0              0              _                 NOUN   NN       Luft              Case=Acc|Gender=Fem|Number=Sing                             app   46    _       O       _               0         Scene        _         _              
48  .                   0              0              _                 PUNCT  $.       .                 PunctType=Peri                                              root  0     _       O       _               0         Scene        _         _              
49  Plötzlich           1              1              _                 ADJ    ADJD     plötzlich         Degree=Pos|Variant=Short                                    adv   50    _       O       _               1         Scene        3         process        
50  nahmen              0              0              _                 VERB   VVFIN    nahmen            Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin       root  0     _       O       _               1         Scene        3         process        
51  aber                0              0              _                 ADV    ADV      aber              _                                                           adv   50    _       O       _               1         Scene        3         process        
52  alle                0              0              _                 DET    PIAT     aller             Case=Nom|Gender=Neut|Number=Plur|PronType=Ind,Neg,Tot       det   54    _       O       _               1         Scene        3         process        
53  diese               0              0              _                 DET    PDAT     dieser            Case=Nom|Gender=Neut|Number=Plur|PronType=Dem               det   54    _       O       _               1         Scene        3         process        
54  Verhältnisse        0              0              _                 NOUN   NN       Verhältnis        Case=Nom|Gender=Neut|Number=Plur                            subj  50    _       O       _               1         Scene        3         process        
55  eine                0              0              _                 DET    ART      ein               Case=Acc|Gender=Fem|Number=Sing|PronType=Art                det   57    _       O       _               1         Scene        3         process        
56  neue                0              0              _                 ADJ    ADJA     neu               Case=Acc|Degree=Pos|Gender=Fem|Number=Sing                  attr  57    _       O       _               1         Scene        3         process        
57  Gestalt             0              0              _                 NOUN   NN       Gestalt           Case=Acc|Gender=Fem|Number=Sing                             obja  50    _       O       _               1         Scene        3         process        
58  an                  0              0              _                 ADP    PTKVZ    an                PartType=Vbp                                                avz   50    _       O       _               1         Scene        3         process        
59  .                   0              0              _                 PUNCT  $.       .                 PunctType=Peri                                              root  0     _       O       _               1         Scene        3         process        
```

Fields
* `i`: Index of the token, counted through the document, starting at 0.
* `text`: Text representation / word form of the token.
* `is_sent_start`: takes on value `1` if this token starts a new sentence, cf. [SoMaJoTokenizer](./DEVELOPING.md#tokenization).
* `is_para_start`: takes on value `1` if this token starts a new paragraph, cf. [SoMaJoTokenizer](./DEVELOPING.md#tokenization).
* `is_section_start`: takes on value `1` if this token starts a new paragraph, cf. [SoMaJoTokenizer](./DEVELOPING.md#tokenization).
* `tag`: POS tag of the token from the [TIGER variant of the STTS tagset](https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/tiger-corpus/annotation/tiger_scheme-syntax.pdf) ([Overview](https://www.linguistik.hu-berlin.de/de/institut/professuren/korpuslinguistik/mitarbeiter-innen/hagen/STTS_Tagset_Tiger)), cf. component [tagger_someweta](./DEVELOPING.md#pos-tagging)
* `pos`: Part-of-Speech tag of the token from the [Universal Dependencies v2 POS tagset](https://universaldependencies.org/u/pos/all.html), automatically converted using table [*de::stts*](https://universaldependencies.org/tagset-conversion/de-stts-uposf.html)
* `lemma`: Lemma of the word form, cf. component [lemma_rnntagger](./DEVELOPING.md#lemmatization).
* `morph`: Morphological features in [CONNL-U format](https://universaldependencies.org/format.html#morphological-annotation), with features from the [Universal features inventory](https://universaldependencies.org/u/feat/index.html). Cf. component [lemma_rnntagger](./DEVELOPING.md#morphological-analysis).
* `head`: Head of the current word, encoded by the head's value `i`. For roots, this is identical to the token's `i`.
* `dep`: Dependency relation of the token's head to this token. Labels are from “Eine umfassende Constraint-Dependenz-Grammatik des Deutschen” ([Forth, 2005](#ref-forth_umfassende_2014)) as labels, used in, e.g., the Hamburg Dependency Treebank ([Overview](https://github.com/rsennrich/ParZu/blob/master/doc/LABELS.md)). Cf. component [parser_parzu](./DEVELOPING.md#dependency-parsing).
* `entity`: NER tag of this token, in [IOB encoding](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)). Uses the usual four classes `PER, LOC, ORG, MISC` as employed by the respective CoNLL-2003 shared task on named entity recognition. ([Sang and De Meulder, 2003](#ref-tjong_kim_sang_introduction_2003)). Cf. component [ner_flair](./DEVELOPING.md#named-entity-recognition).
* `speech`: Comma-separated list of speech types annotated on this token, or `_` if none. A subset of `direct`, `indirect`, `freeIndirect` and `reported`, as defined by [Brunner et al. (2021)](#ref-brunner_bert_2021). Cf. component [speech_redewiedergabe](./DEVELOPING.md#recognition-of-speech-thought-and-writing-representation).
* `coref_clusters`: Comma-separated list of coreference clusters (represented by integer IDs) annotated on this token, or `_` if none. Cf. component [coref_uhhlt](./DEVELOPING.md#coreference-resolution).
* `scene_id`: Numerical index of the scene which contains this token, cf. component [scenes_stss_se](./DEVELOPING.md#scene-segmentation).
* `scene_label`: Annotated label of the scene having index `scene_id`. One of `Scene` or `Nonscene`.
* `event_id`: Numerical index of the event which contains this token, or `_` if token is not contained in an event. Cf. component [events_uhhlt](./DEVELOPING.md#event-classification).
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
