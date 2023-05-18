# Output Formats

In the default configuration, the script `main.py` outputs its result as tabular data in a pseudo-CONLL format.
Each token is represented on one line, with fields separated by tabs.
The entire output is encoded in plain text (UTF-8, normalized to NFC, using only the LF character as line break,
normalized to NFC in the default configuration).
Unlike the traditional CONNL-U format, sentences are not separated by newlines.
Boolean values are encoded as `0`/`1`.

Example (where, for readability, tabs are replaced with spaces to align the columns):
```
i  text             orig             is_sent_start is_para_start is_section_start pos   tag   lemma            morph                                                  dep  head speech          entity character coref_clusters scene_id scene_label event_id event_label  
0  »                »                1             _             _                PUNCT $(    »                PunctType=Brck                                         root 0    direct          O      O         _              0        Scene       _        _            
1  Fürchten         Fürchten         0             _             _                VERB  VVFIN fürchten         Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin  root 1    direct          O      O         _              0        Scene       0        non_event    
2  Sie              Sie              0             _             _                PRON  PPER  sie              Case=Nom|Gender=*|Number=Plur|Person=3|PronType=Prs    subj 1    direct          O      O         0              0        Scene       0        non_event    
3  nichts           nichts           0             _             _                PRON  PIS   nichts           Case=*|Gender=Neut|Number=*|PronType=Ind,Neg,Tot       obja 1    direct          O      O         _              0        Scene       0        non_event    
4  für              für              0             _             _                ADP   APPR  für              AdpType=Prep                                           pp   1    direct          O      O         _              0        Scene       0        non_event    
5  mich             mich             0             _             _                PRON  PPER  mich             Case=Acc|Gender=*|Number=Sing|Person=1|PronType=Prs    pn   4    direct          O      O         0              0        Scene       0        non_event    
6  ,                ,                0             _             _                PUNCT $,    ,                PunctType=Comm                                         root 6    direct          O      O         _              0        Scene       _        _            
7  lieber           lieber           0             _             _                ADV   ADV   lieb             Case=Nom|Degree=Pos|Gender=Masc|Number=Sing            adv  8    direct          O      O         _              0        Scene       0        non_event    
8  Freund           Freund           0             _             _                NOUN  NN    Freund           Case=Acc|Gender=Masc|Number=Sing                       app  5    direct          O      B-PER     0              0        Scene       0        non_event    
9  .                .                0             _             _                PUNCT $.    .                PunctType=Peri                                         root 9    direct          O      O         _              0        Scene       _        _            
10 Ich              Ich              1             _             _                PRON  PPER  ich              Case=Nom|Gender=*|Number=Sing|Person=1|PronType=Prs    subj 11   direct,reported O      O         0              0        Scene       1        process      
11 glaube           glaube           0             _             _                VERB  VVFIN glauben          Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin  root 11   direct,reported O      O         _              0        Scene       1        process      
12 an               an               0             _             _                ADP   APPR  an               AdpType=Prep                                           objp 11   direct,reported O      O         _              0        Scene       1        process      
13 das              das              0             _             _                DET   ART   der              Case=Acc|Gender=Neut|Number=Sing|PronType=Art          det  14   direct,reported O      O         _              0        Scene       1        process      
14 Geheimnis        Geheimnis        0             _             _                NOUN  NN    Geheimnis        Case=Acc|Gender=Neut|Number=Sing                       pn   12   direct,reported O      O         _              0        Scene       1        process      
15 ,                ,                0             _             _                PUNCT $,    ,                PunctType=Comm                                         root 15   direct,reported O      O         _              0        Scene       _        _            
16 das              das              0             _             _                PRON  PRELS der              Case=Acc|Gender=Neut|Number=Sing|PronType=Rel          obja 20   direct,reported O      O         _              0        Scene       1        process      
17 er               er               0             _             _                PRON  PPER  er               Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs subj 21   direct,reported O      O         0              0        Scene       1        process      
18 mit              mit              0             _             _                ADP   APPR  mit              AdpType=Prep                                           pp   20   direct,reported O      O         _              0        Scene       1        process      
19 sich             sich             0             _             _                PRON  PRF   sich             Case=Dat|Number=Sing|Person=3|PronType=Prs|Reflex=Yes  pn   18   direct          O      O         0              0        Scene       1        process      
20 genommen         genommen         0             _             _                VERB  VVPP  nehmen           Aspect=Perf|VerbForm=Part                              aux  21   direct,reported O      O         _              0        Scene       1        process      
21 hat              hat              0             _             _                AUX   VAFIN haben            Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin  rel  14   direct,reported O      O         _              0        Scene       1        process      
22 .                .                0             _             _                PUNCT $.    .                PunctType=Peri                                         root 22   direct,reported O      O         _              0        Scene       _        _            
23 Ich              Ich              1             _             _                PRON  PPER  ich              Case=Nom|Gender=*|Number=Sing|Person=1|PronType=Prs    subj 24   direct          O      O         0              0        Scene       2        non_event    
24 werde            werde            0             _             _                AUX   VAFIN werden           Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin  root 24   direct          O      O         _              0        Scene       2        non_event    
25 weiter           weiter           0             _             _                ADV   ADV   weiter           _                                                      adv  28   direct          O      O         _              0        Scene       2        non_event    
26 den              den              0             _             _                DET   ART   der              Case=Acc|Gender=Masc|Number=Sing|PronType=Art          det  27   direct          O      O         _              0        Scene       2        non_event    
27 Menschen         Menschen         0             _             _                NOUN  NN    Mensch           Case=Acc|Gender=Masc|Number=Sing                       objd 28   direct          O      B-PER     1              0        Scene       2        non_event    
28 dienen           dienen           0             _             _                VERB  VVINF dienen           VerbForm=Inf                                           aux  24   direct          O      O         _              0        Scene       2        non_event    
29 .                .                0             _             _                PUNCT $.    .                PunctType=Peri                                         root 29   direct          O      O         _              0        Scene       2        non_event    
30 «                «                1             _             _                PUNCT $(    «                PunctType=Brck                                         root 30   direct          O      O         _              1        Scene       _        _            
31 Unterwegs        Unterwegs        0             _             _                ADV   ADV   unterwegs        _                                                      adv  32   reported        O      O         _              1        Scene       3        process      
32 erreichte        erreichte        0             _             _                VERB  VVFIN erreichen        Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin  root 32   reported        O      O         _              1        Scene       3        process      
33 Konrad           Konrad           0             _             _                PROPN NE    Konrad           Case=Nom|Gender=Masc|Number=Sing                       subj 32   reported        B-PER  B-PER     0              1        Scene       3        process      
34 Hochseß          Hochseß          0             _             _                PROPN NE    Hochseß          Case=Nom|Gender=Masc|Number=Sing                       app  33   reported        I-PER  I-PER     0              1        Scene       3        process      
35 die              die              0             _             _                DET   ART   der              Case=Acc|Gender=Fem|Number=Sing|PronType=Art           det  36   reported        O      O         _              1        Scene       3        process      
36 Nachricht        Nachricht        0             _             _                NOUN  NN    Nachricht        Case=Acc|Gender=Fem|Number=Sing                        obja 32   reported        O      O         _              1        Scene       3        process      
37 der              der              0             _             _                DET   ART   der              Case=Gen|Gender=Fem|Number=Sing|PronType=Art           det  39   reported        O      O         _              1        Scene       3        process      
38 schweren         schweren         0             _             _                ADJ   ADJA  schwer           Case=Gen|Degree=Pos|Gender=Fem|Number=Sing             attr 39   reported        O      O         _              1        Scene       3        process      
39 Erkrankung       Erkrankung       0             _             _                NOUN  NN    Erkrankung       Case=Gen|Gender=Fem|Number=Sing                        gmod 36   reported        O      O         _              1        Scene       3        process      
40 der              der              0             _             _                DET   ART   der              Case=Gen|Gender=Fem|Number=Sing|PronType=Art           det  41   reported        O      O         _              1        Scene       3        process      
41 Gräfin           Gräfin           0             _             _                NOUN  NN    Gräfin           Case=Gen|Gender=Fem|Number=Sing                        gmod 39   reported        O      B-PER     2              1        Scene       3        process      
42 Savelli          Savelli          0             _             _                PROPN NE    Savelli          Case=Nom|Gender=Fem|Number=Sing                        app  41   reported        B-PER  I-PER     2              1        Scene       3        process      
43 .                .                0             _             _                PUNCT $.    .                PunctType=Peri                                         root 43   reported        O      O         _              1        Scene       _        _            
44 Auf              Auf              1             _             _                ADP   APPR  auf              AdpType=Prep                                           root 44   _               O      O         _              1        Scene       _        _            
45 der              der              0             _             _                DET   ART   der              Case=Dat|Gender=Fem|Number=Sing|PronType=Art           det  46   _               O      O         _              1        Scene       _        _            
46 Chaussee         Chaussee         0             _             _                NOUN  NN    Chaussee         Case=Dat|Gender=Fem|Number=Sing                        pn   44   _               O      O         _              1        Scene       _        _            
47 von              von              0             _             _                ADP   APPR  von              AdpType=Prep                                           pp   46   _               O      O         _              1        Scene       _        _            
48 Hochseß          Hochseß          0             _             _                PROPN NE    Hochseß          Case=Dat|Gender=Masc|Number=Sing                       pn   47   _               B-LOC  O         0              1        Scene       _        _            
49 nach             nach             0             _             _                ADP   APPR  nach             AdpType=Prep                                           kon  47   _               O      O         _              1        Scene       _        _            
50 Ebermannstadt    Ebermannstadt    0             _             _                PROPN NE    Ebermannstadt    Case=Dat|Gender=Neut|Number=Sing                       pn   49   _               B-LOC  O         _              1        Scene       _        _            
51 –                –                0             _             _                PUNCT $(    –                AdpType=Prep|PunctType=Brck                            root 51   _               O      O         _              1        Scene       _        _            
52 der              der              0             _             _                DET   ART   der              Case=Dat|Gender=Fem|Number=Sing|PronType=Art           det  54   _               O      O         _              1        Scene       _        _            
53 nächsten         nächsten         0             _             _                ADJ   ADJA  nächster         Case=Dat|Degree=Pos|Gender=Fem|Number=Sing             attr 54   _               O      O         _              1        Scene       _        _            
54 Eisenbahnstation Eisenbahnstation 0             _             _                NOUN  NN    Eisenbahnstation Case=Dat|Gender=Fem|Number=Sing                        gmod 50   _               O      O         _              1        Scene       _        _            
55 –                –                0             _             _                PUNCT $(    –                PunctType=Brck                                         root 55   _               O      O         _              1        Scene       _        _            
56 die              die              0             _             _                PRON  PRELS der              Case=Nom|Gender=Fem|Number=Sing|PronType=Rel           subj 61   _               O      O         _              1        Scene       4        stative_event
57 über             über             0             _             _                ADP   APPR  über             AdpType=Prep                                           pp   61   _               O      O         _              1        Scene       4        stative_event
58 das              das              0             _             _                DET   ART   der              Case=Acc|Gender=Neut|Number=Sing|PronType=Art          det  60   _               O      O         _              1        Scene       4        stative_event
59 kahle            kahle            0             _             _                ADJ   ADJA  kahl             Case=Acc|Degree=Pos|Gender=Neut|Number=Sing            attr 60   _               O      O         _              1        Scene       4        stative_event
60 Hochplateau      Hochplateau      0             _             _                NOUN  NN    Hochplateau      Case=Acc|Gender=Neut|Number=Sing                       pn   57   _               O      O         _              1        Scene       4        stative_event
61 hinüberführte    hinüberführte    0             _             _                VERB  VVFIN hinüberführen    Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin  root 61   _               O      O         _              1        Scene       4        stative_event
62 ,                ,                0             _             _                PUNCT $,    ,                PunctType=Comm                                         root 62   _               O      O         _              1        Scene       _        _            
63 standen          standen          0             _             _                VERB  VVFIN stehen           Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin  root 63   _               O      O         _              1        Scene       5        process      
64 kleine           kleine           0             _             _                ADJ   ADJA  klein            Case=Nom|Degree=Pos|Gender=Fem|Number=Plur             attr 67   _               O      O         _              1        Scene       5        process      
65 ,                ,                0             _             _                PUNCT $,    ,                PunctType=Comm                                         root 65   _               O      O         _              1        Scene       _        _            
66 schmutzige       schmutzige       0             _             _                ADJ   ADJA  schmutzig        Case=Nom|Degree=Pos|Gender=Fem|Number=Plur             kon  64   _               O      O         _              1        Scene       5        process      
67 Wasserlachen     Wasserlachen     0             _             _                NOUN  NN    Wasserlache      Case=Nom|Gender=Fem|Number=Plur                        subj 63   _               O      O         _              1        Scene       5        process      
68 .                .                0             _             _                PUNCT $.    .                PunctType=Peri                                         root 68   _               O      O         _              1        Scene       _        _            
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
* `dep`: Dependency relation of the token's head to this token. Labels are from “Eine umfassende Constraint-Dependenz-Grammatik des Deutschen” ([Foth, 2005](#ref-foth_umfassende_2014)) as labels, used in, e.g., the Hamburg Dependency Treebank ([Overview](https://github.com/rsennrich/ParZu/blob/master/doc/LABELS.md)). Cf. component [parser_parzu](./DEVELOPING.md#dependency-parsing).
* `entity`: NER tag of this token, in [IOB encoding](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)). Uses the usual four classes `PER, LOC, ORG, MISC` as employed by the respective CoNLL-2003 shared task on named entity recognition. ([Sang and De Meulder, 2003](#ref-tjong_kim_sang_introduction_2003)). Cf. component [ner_flair](./DEVELOPING.md#named-entity-recognition).
* `character`: Character Mention tag of this token, in [IOB encoding](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)). Uses only `B-PER` and `I-PER` to annotate character mentions. ([Krug et al., 2017](#ref-krug_description_2017)). Cf. component [character_recognizer](./DEVELOPING.md#character-recognizer).
* `speech`: Comma-separated list of speech types annotated on this token, or `_` if none. A subset of `direct`, `indirect`, `freeIndirect` and `reported`, as defined by [Brunner et al. (2021)](#ref-brunner_bert_2021). Cf. component [speech_redewiedergabe](./DEVELOPING.md#recognition-of-speech-thought-and-writing-representation).
* `coref_clusters`: Comma-separated list of coreference clusters (represented by integer IDs) annotated on this token, or `_` if none. Cf. component [coref_uhhlt](./DEVELOPING.md#coreference-resolution).
* `scene_id`: Numerical index of the scene which contains this token, cf. component [su_scene_segmenter](./DEVELOPING.md#scene-segmentation).
* `scene_label`: Annotated label of the scene having index `scene_id`. One of `Scene` or `Nonscene`.
* `event_id`: Numerical index of the event which contains this token, or `_` if token is not contained in an event. Cf. component [events_uhhlt](./DEVELOPING.md#event-classification).
* `event_label`: Annotated label of the event having index `event_id`. One of `non_event`, `change_of_state`, `process` or `stative_event`, as defined by [Vauth et al. (2021)](#ref-vauth_automated_2021).

# References

<div id="ref-brunner_bert_2021">

<p>Brunner, Annelen, Ngoc Duyen Tanja Tu, Lukas Weimer, and Fotis Jannidis. 2021. “To BERT or Not to BERT – Comparing Contextual Embeddings in a Deep Learning Architecture for the Automatic Recognition of Four Types of Speech, Thought and Writing Representation.” In <em>Proceedings of the 5th Swiss Text Analytics Conference (SwissText) &amp; 16th Conference on Natural Language Processing (KONVENS)</em>, 2624:11. CEUR Workshop Proceedings. Zurich, Switzerland. <a href="http://ceur-ws.org/Vol-2624/paper5.pdf">http://ceur-ws.org/Vol-2624/paper5.pdf</a>.</p>

</div>

<div id="ref-foth_umfassende_2014" class="csl-entry" role="doc-biblioentry">

Foth, Kilian A. 2014. <em>Eine Umfassende Constraint-Dependenz-Grammatik Des Deutschen</em>. Universität Hamburg. <a href="https://edoc.sub.uni-hamburg.de/informatik/volltexte/2014/204/">https://edoc.sub.uni-hamburg.de/informatik/volltexte/2014/204/</a>.

</div>

<div id="ref-krug_description_2017">

<p>Krug, Markus, Lukas Weimer, Isabella Reger, Luisa Macharowsky, Stephan Feldhaus, Frank Puppe, and Fotis Jannidis. 2017. “Description of a Corpus of Character References in German Novels - DROC [Deutsches ROman Corpus].” <a href="https://resolver.sub.uni-goettingen.de/purl?gro-2/108301">https://resolver.sub.uni-goettingen.de/purl?gro-2/108301</a>.</p>

</div>

<div id="ref-tjong_kim_sang_introduction_2003" class="csl-entry" role="doc-biblioentry">

Sang, Erik F. Tjong Kim, and Fien De Meulder. 2003. <span>“Introduction to the <span>CoNLL</span>-2003 Shared Task: Language-Independent Named Entity Recognition.”</span> In <em>Proceedings of the Seventh Conference on Natural Language Learning at <span>HLT</span>-<span>NAACL</span> 2003</em>, 142–47. Edmonton, Canada: Association for Computational Linguistics. <a href="https://doi.org/10.3115/1119176.1119195">https://doi.org/10.3115/1119176.1119195</a>.

</div>

<div id="ref-vauth_automated_2021">

<p>Vauth, Michael, Hans Ole Hatzel, Evelyn Gius, and Chris Biemann. 2021. “Automated Event Annotation in Literary Texts.” In <em>Proceedings of the Conference on Computational Humanities Research 2021</em>, edited by Maud Ehrmann, Folgert Karsdorp, Melvin Wevers, Tara Lee Andrews, Manuel Burghardt, Mike Kestemont, Enrique Manjavacas, Michael Piotrowski, and Joris van Zundert, 2989:333–45. CEUR Workshop Proceedings. Amsterdam, the Netherlands. <a href="https://ceur-ws.org/Vol-2989/#short_paper18">https://ceur-ws.org/Vol-2989/#short_paper18</a>.</p>

</div>
