# Developing

The LLpro pipeline is implemented using the Spacy framework. In particular, this repository implements several [custom pipeline components](https://spacy.io/usage/processing-pipelines#custom-components) to construct full Spacy pipelines.
For example, the default implementation in `main.py` constructs a Spacy pipeline as follows:

```python
import spacy
from llpro.components import *

# implement a blank pipeline
nlp = spacy.blank("de")

# add the custom components
nlp.add_pipe('tagger_someweta')
nlp.add_pipe('tagger_rnntagger')
nlp.add_pipe('lemma_rnntagger')
nlp.add_pipe('parser_parzu_parallelized')
nlp.add_pipe('speech_redewiedergabe')
nlp.add_pipe('su_scene_segmenter')
nlp.add_pipe('coref_uhhlt')
nlp.add_pipe('ner_flair')
nlp.add_pipe('events_uhhlt')

# replace the default tokenizer with custom one
import llpro.components.tokenizer_somajo
nlp.tokenizer = llpro.components.tokenizer_somajo.SoMaJoTokenizer(nlp.vocab)

# run the pipeline on some example text
doc = nlp("Das ist der Beginn eines literarischen Textes.")

# after processing, we can examine the annotations on the tokens, e.g. as follows:
for token in doc:
    print(token.i, token.text, token.tag_)
```

Note that you can pass additional options by specifying them in the `config` dict when adding to the pipeline:
```python
nlp.add_pipe('the_component', config={'some_option': False})
```

## Common Options of the Custom Components

The following options can be passed to many of the custom components below, and are specified here:

| Name            | Description                                                                                                                                                                                                                                                                     |
|:----------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `use_cuda`      | If `True`, use CUDA to run the component on the GPU, if available Default: `True`                                                                                                                                                                                               |
| `device_on_run` | If `True`, will load the component's model(s) into the GPU memory only when the component is being called. If `False`, the component will load the model(s) into GPU memoty immediately after the component is added to the pipeline. Default: `False`                          |
| `pbar_opts`     | Dict object that is passed to the [tqdm progress bar constructior](https://tqdm.github.io/docs/tqdm/#__init__) as keywords, which is used to display progress of the component when being called. Default: `{'unit': 'tok', 'postfix': self.name, 'ncols': 80, 'leave': False}` |


## List of Implemented Custom Components / Tokenizer

### Tokenization

* Implementing class: `llpro.components.tokenizer_somajo.SoMaJoTokenizer`.

The SoMaJoTokenizer uses the tokenizer [SoMaJo](https://github.com/tsproisl/SoMaJo) proposed by Proisl and Uhrig [(2016)](#ref-proisl_somajo_2016)
to perform tokenization and sentence splitting. Additionally, like Spacy's [sentence segmenters](https://spacy.io/usage/linguistic-features#sbd), it assigns the attribute `token.is_sent_start` and, after tokenization, `doc.sents` enumerates all sentences of the document.
Additionally, it assigns the original unnormalized token to the attribute `token._.orig`, and the character offset of its occurrence in the original input text to the attribute `token._.orig_offset`.

Additionally, the constructor takes the following optional keyword arguments:

| Name                  | Description                                                                                                                                                                                                                                                                                  |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `normalize`           | Before tokenization, normalizes the input text by replacing U+0364 Combining Latin Small Letter E with U+0308 Combining Diaeresis, followed by conversion into NFKC normal form. Default: `True`                                                                                             |
| `check_characters`    | Warn to stderr if the tokenizer encounters “unusual” characters (mostly non-Latin, non-punctuation characters). Default: `True`                                                                                                                                                              |
| `paragraph_separator` | If not `None`, the value is interpreted as regular expression and the text is split at every match. The matching spans are discarded. Paragraphs starts are stored at the custom attribute `token._.is_para_start`. Default: `None`                                                          |
| `section_pattern`     | If not `None`, the value is interpreted as regular expression. If a paragraph fully matches the pattern, the paragraph is interpreted as section start. The matching paragraphs are discarded. Section starts are stored at the custom attribute `token._.is_section_start`. Default: `None` |
| `is_pretokenized`     | If `True`, skip tokenization, and assume that tokens are separated by whitespace. Default: `False`                                                                                                                                                                                           |
| `is_presentencized`   | If `True`, skip sentence splitting, and assume that sentences are separated by newline characters. Default: `False`                                                                                                                                                                          |

In the default implementation `main.py`, arguments `normalize`, `paragraph_separator`, `section_pattern` `is_pretokenized`, `is_presentencized` are supplied by the command-line arguments.

### POS Tagging

* Component name: `tagger_someweta`
* Implementing class: `llpro.components.tagger_someweta.SoMeWeTaTagger`
* Assigns: `token.pos_`, `token.tag_`
 
Options:

| Name            | Description                                                                   |
|-----------------|:------------------------------------------------------------------------------|
| `model`         | SoMeWeTa model to use. Default: `resources/german_newspaper_2020-05-28.model` |
| `pbar_opts`     | as specified above                                                            |

The SoMeWeTaTagger component uses the POS tagger [SoMeWeTa](https://github.com/tsproisl/SoMeWeTa) proposed by Proisl [(2018)](#ref-proisl_someweta_2018)
to predict POS tags ([TIGER variant of the STTS tagset](https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/tiger-corpus/annotation/tiger_scheme-syntax.pdf)) for each token.

Like Spacy's [default tagger](https://spacy.io/usage/linguistic-features#pos-tagging), it assigns the STTS POS tag to attribute `token.tag_`, and it assigns a POS tag from the [Universal Dependencies v2 POS tagset](https://universaldependencies.org/u/pos/all.html) to the attribute `token.pos_`.
(automatically converted using table [*de::stts*](https://universaldependencies.org/tagset-conversion/de-stts-uposf.html)).

### Morphological Analysis

* Component name: `tagger_rnntagger`
* Implementing class: `llpro.components.tagger_rnntagger.RNNTagger`
* Assigns: `token._rnntagger_tag`, `token.morph`

Options:

| Name             | Description                                                                                                                                                  |
|------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `rnntagger_home` | Root directory of the RNNTagger, i.e. where `PyNMT`, `PyRNN` resides. (Adaptations of the code are not required in this case.) Default: `resoures/RNNTagger` |
| `use_cuda`       | as specified above                                                                                                                                           |
| `device_on_run`  | as specified above                                                                                                                                           |
| `pbar_opts`      | as specified above                                                                                                                                           |

The RNNTagger component uses the POS tagger and Analyzer [RNNTagger](https://www.cis.uni-muenchen.de/~schmid/tools/RNNTagger/) proposed by Schmid [(2019)](#ref-schmid_deep_2019)
to predict for each token a morphological analysis and an additional POS token for further processing.

Like Spacy's [default morphological analyzer](https://spacy.io/usage/linguistic-features#morphology), it assigns a `MorphAnalysis` object to the attribute `token.morph`.
Additionally, the predicted POS tag is stored in the custom attribute `token._.rnntagger_tag`. This second attribute is only used for further processing by RNNTagger's lemmatizer (`lemma_rnntagger`).

### Lemmatization

* Component name: `lemma_rnntagger`
* Implementing class: `llpro.components.lemma_rnntagger.RNNLemmatizer`
* Assigns: `token.lemma_`
* Requires: `token._rnntagger_tag`

Options:

| Name             | Description                                                                                                                                                  |
|:-----------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `rnntagger_home` | Root directory of the RNNTagger, i.e. where `PyNMT`, `PyRNN` resides. (Adaptations of the code are not required in this case.) Default: `resoures/RNNTagger` |
| `use_cuda`       | as specified above                                                                                                                                           |
| `device_on_run`  | as specified above                                                                                                                                           |
| `pbar_opts`      | as specified above                                                                                                                                           |

The RNNLemmatizer component uses the Lemmatizer [RNNTagger](https://www.cis.uni-muenchen.de/~schmid/tools/RNNTagger/) proposed by Schmid [(2019)](#ref-schmid_deep_2019)
to predict for each token the lemma form using the POS tag predicted by the same RNNTagger (`token._rnntagger_tag_`).

Like Spacy's [default lemmatizer](https://spacy.io/usage/linguistic-features#lemmatization), it assigns the lemma form to the attribute `token.lemma_`.

### Depencency Parsing

* Component name: `parser_parzu_parallelized`
* Implementing class: `llpro.components.parser_parzu.ParzuParallelized`
* Assigns: `token.head`, `token.dep_`
* Requires: `token.tag_`

Options:

| Name                 | Description                                                                                              |
|:---------------------|:---------------------------------------------------------------------------------------------------------|
| `parzu_home`         | Root directory of the ParZu parser *adapted for use with the LLpro pipeline*. Default: `resources/ParZu` |
| `num_processes`      | Number of parallel processes to run the parser on. Default: `1`                                          |
| `tokens_per_process` | Max chunk size per process. Default: `1000`                                                              |
| `pbar_opts`          | as specified above                                                                                       |

The ParzuParallelized component uses the dependency parser [ParZu](https://github.com/rsennrich/ParZu) proposed by Sennrich, Schneider, Volk, and Warin [(2009)](#ref-sennrich_new_2009), Sennrich, Volk, and Schneider [(2013)](#ref-sennrich_exploiting_2013), and Sennrich and Kunz [(2014)](#ref-sennrich_zmorge_2014)
to predict for each (non-punctuation) token a head token and a dependency relation.
In particular, it follows the grammar “Eine umfassende Constraint-Dependenz-Grammatik des Deutschen” ([Foth, 2005](#ref-foth_umfassende_2014); [Overview](https://github.com/rsennrich/ParZu/blob/master/doc/LABELS.md)).

Like Spacy's [default dependency parser](https://spacy.io/usage/linguistic-features#dependency-parse), it assigns to each token a pointer `tok.head` to its head token, and the dependency relation label to `tok.dep_`.
As usual, you can use Spacy's API to navigate the resulting parse tree, e.g. enumerate noun chunks with `doc.noun_chunks`.

Unlike Spacy's default dependency parser, this component does not modify the sentence segmentation and retains the segmentation present before the parser is run.

### Named Entity Recognition

* Component name: `ner_flair`
* Implementing class: `llpro.components.ner_flair.FLERTNERTagger`
* Assigns: `doc.ents`, `token.ent_iob_`, `token.ent_type_`

Options:

| Name            | Description                                                                |
|:----------------|:---------------------------------------------------------------------------|
| `model`         | Flair NER model to use. Default: `flair/ner-german-large`                  |
| `batch_size`    | Number of sentences concurrently processed by one prediction. Default: `8` |
| `use_cuda`      | as specified above                                                         |
| `device_on_run` | as specified above                                                         |
| `pbar_opts`     | as specified above                                                         |

The FLERTNERTagger component uses the NER recognizer [FLERT](https://github.com/flairNLP/flair) (part of the Flair NLP system) proposed by Schweter and Akbik [(2021)](#ref-schweter_flert_2021),
to predict named entities in the document.
It uses the usual four classes `PER, LOC, ORG, MISC` (in IOB encoding) as employed by the respective CoNLL-2003 shared task on named entity recognition ([Sang and De Meulder, 2003](#ref-tjong_kim_sang_introduction_2003)).

Like Spacy's [default entity recognizer](https://spacy.io/usage/linguistic-features#named-entities), it assigns to the document to the attribute `doc.ents` a list of named entities, and to each token also assigns the IOB code to the attribute `tok.ent_iob_` and the entity type label to the attribute `tok.ent_type_`.

### Character Recognizer

* Component name: `character_recognizer`
* Implementing class: `llpro.components.character_recognizer.CharacterRecognizer`
* Assigns: `doc._.characters`, `token._.character_iob`

Options:

| Name            | Description                                                                |
|:----------------|:---------------------------------------------------------------------------|
| `model`         | Flair NER model to use. Default: `aehrm/droc-character-recognizer`         |
| `batch_size`    | Number of sentences concurrently processed by one prediction. Default: `8` |
| `use_cuda`      | as specified above                                                         |
| `device_on_run` | as specified above                                                         |
| `pbar_opts`     | as specified above                                                         |

The CharacterRecognizer component uses a custom Flair model as proposed Schweter and Akbik [(2021)](#ref-schweter_flert_2021), fine-tuned for the DROC dataset, to recognize references to literary characters.
It uses only the single class `PER` (in IOB encoding) to denote mentions of literary characters [(Krug et al., 2017)](#ref-krug_description_2017).

Similar to Spacy's [default entity recognizer](https://spacy.io/usage/linguistic-features#named-entities), it assigns to the document to the attribute `doc._.characters` a list of named entities, and to each token also assigns the IOB code to the attribute `tok._.character_iob` (one of `I`, `O`, `B`).

### Coreference Resolution

* Component name: `coref_uhhlt`
* Implementing class: `llpro.components.coref_uhhlt.CorefTagger`
* Assigns: `token._.in_coref`, `token._.coref_clusters`, `doc._.has_coref`, `doc._.coref_clusters`

Options:

| Name            | Description                                                                                                                                                                                                                                                                                                                                |
|:----------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `coref_home`    | Root directory of the local neural-coref repository *adapted for use with the LLpro pipeline*. Default: `resources/uhh-lt-neural-coref`.                                                                                                                                                                                                   |
| `config_name`   | Model configuration to use. See also the respective documentation in the [UHH-LT/neural-coref](https://github.com/uhh-lt/neural-coref/tree/konvens#configurations) repository. Default: `droc_incremental_no_segment_distance`                                                                                                             |
| `model`         | Model state dict to load. Should match the specified model configuration. Usually ends with `.bin`. See also the respective documentation in the [UHH-LT/neural-coref](https://github.com/uhh-lt/neural-coref/tree/konvens#evaluation) repository. Default: `resources/model_droc_incremental_no_segment_distance_May02_17-32-58_1800.bin` |
| `use_cuda`      | as specified above                                                                                                                                                                                                                                                                                                                         |
| `device_on_run` | as specified above                                                                                                                                                                                                                                                                                                                         |
| `pbar_opts`     | as specified above                                                                                                                                                                                                                                                                                                                         |

The CorefTagger component uses the neural coreference tagger proposed by Schröder, Hatzel and Biemann [(2021)](#ref-schroder_neural_2021) to predict coreference clusters in the document.
The default implementation uses the *incremental* variant of their model which is capable of performing coreference resolution even on longer documents.

This component assigns custom attributes similar to huggingface's custom Spacy component [NeuralCoref](https://github.com/huggingface/neuralcoref).
Note that the assigned attributes differ from Spacy's [experimental implementation of a coreference resolver](https://spacy.io/api/coref).

In particular, it assigns the following annotations to the document and to tokens:

| Attribute                | Type                | Description                                                 |
|--------------------------|---------------------|-------------------------------------------------------------|
| `doc._.has_coref`        | boolean             | Has any coreference has been resolved in the Doc.           |
| `doc._.coref_clusters`   | list of `SpanGroup` | A list of recognized coreference clusters, described below. |
| `token._.in_coref`       | boolean             | Is this token present in at least one coreference cluster.  |
| `token._.coref_clusters` | list of `SpanGroup` | A list of coreference clusters this token is present in.    |

The coreference clusters are represented by Spacy's [`SpanGroup`](https://spacy.io/api/spangroup).
For every coreference custer, `span_group.spans` contains a list of spans that represent the mentions of that cluster.
Additionally, `span_group.attrs["id"]` holds a unique ID of that cluster.


### Recognition of Speech, Thought, and Writing Representation

* Component name: `speech_redewiedergabe`
* Implementing class: `llpro.components.speech_redewiedergabe.RedewiedergabeTagger`
* Assigns: `token._.speech`

Options:

| Name            | Description                                                                                                                                                                                                                                                                                            |
|:----------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `models`        | A dict with keys corresponding to the STWR type, and values corresponding to the respective model. Default: `{'direct': 'aehrm/redewiedergabe-direct', 'indirect': 'aehrm/redewiedergabe-indirect', 'reported': 'aehrm/redewiedergabe-reported', 'freeIndirect': 'aehrm/redewiedergabe-freeindirect'}` |
| `use_cuda`      | as specified above                                                                                                                                                                                                                                                                                     |
| `device_on_run` | as specified above                                                                                                                                                                                                                                                                                     |
| `pbar_opts`     | as specified above                                                                                                                                                                                                                                                                                     |

The RedewiedergabeTagger uses neural taggers proposed by Brunner, Tu, Weimer, and Jannidis [(2020)](#ref-brunner_bert_2021) to predict for each token four different types of speech, thought and writing representation (STWR).

This component assigns to each token a subset of the four speech types `direct`, `indirect`, `reported`, `freeIndirect` to the custom attribute `token._.speech` for which the respective model labeled the token as part of the respective type of STWR.

### Scene Segmentation

* Component name: `scene_segmenter`
* Implementing class: `llpro.components.scene_segmenter.SceneSegmenter`
* Assigns: `doc._.scenes`, `token._.scene`

Options:

| Name            | Description                                                             |
|:----------------|:------------------------------------------------------------------------|
| `model`         | The scene segmenter model to use. Default: `aehrm/stss-scene-segmenter` |
| `use_cuda`      | as specified above                                                      |
| `device_on_run` | as specified above                                                      |
| `pbar_opts`     | as specified above                                                      |

The SceneSegmenter uses a neural classifier inspired by Kurfalı and Wirén [(2021)](#ref-kurfali_breaking_2021) to predict after each sentence if a new scene/non-scene begins.

This component assigns to the document a list of [`Span`](https://spacy.io/api/span) objects to the custom attribute `doc._.scenes`, where each span represents one scene/non-scene.
For each span `scene` in `doc._.scenes`, the attribute `scene.label_` is one of `'Scene'` or `'Nonscene'`.

Additionally, this component also assigns to each token the containing span to the custom attribute `token._.scene`, again represented by a `Span` object as defined above.


### Event Classification

* Component name: `events_uhhlt`
* Implementing class: `llpro.components.events_uhhlt.EventClassifier`
* Requires: `token.tag`, `token.dep`, `token.head`
* Assigns `doc._.events`

Options:

| Name                  | Description                                                                                                                                  |
|:----------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|
| `event_classify_home` | Root directory of the local event-classify repository *adapted for use with the LLpro pipeline*. Default: `resources/uhh-lt-event-classify`. |
| `model_dir`           | The scene segmenter model to use. Default: `resources/eventclassifier_model/demo_model`                                                      |
| `batch_size`          | Number of sentences concurrently processed by one prediction. Default: `8`                                                                   |
| `use_cuda`            | as specified above                                                                                                                           |
| `device_on_run`       | as specified above                                                                                                                           |
| `pbar_opts`           | as specified above                                                                                                                           |

The EventClassifier uses a neural classifier proposed by Vauth, Hatzel, Gius, and Biemann [(2021)](#ref-vauth_automated_2021) to annotate every verbal phrase with one of four event types `change_of_state`, `process_event`, `stative_event` and `non_event`.

This component assigns to the document a list of [`SpanGroup`](https://spacy.io/api/spangroup) objects to the custom attribute `doc._.events`, where each span group represents one verbal phrase.
For each span group `event` in `doc._.events`, the attribute `event.spans` holds a list of spans that constitute the verbal phrase.
The value `event.attrs["event_type"]` holds the annotated event type, i.e. one of `change_of_state`, `process_event`, `stative_event`, `non_event`.

## References

<div id="ref-akbik_contextual_2018">

<p>Akbik, Alan, Duncan Blythe, and Roland Vollgraf. 2018. “Contextual String Embeddings for Sequence Labeling.” In <em>COLING 2018, 27th International Conference on Computational Linguistics</em>, 1638–49.</p>

</div>

<div id="ref-brunner_bert_2021">

<p>Brunner, Annelen, Ngoc Duyen Tanja Tu, Lukas Weimer, and Fotis Jannidis. 2021. “To BERT or Not to BERT – Comparing Contextual Embeddings in a Deep Learning Architecture for the Automatic Recognition of Four Types of Speech, Thought and Writing Representation.” In <em>Proceedings of the 5th Swiss Text Analytics Conference (SwissText) &amp; 16th Conference on Natural Language Processing (KONVENS)</em>, 2624:11. CEUR Workshop Proceedings. Zurich, Switzerland. <a href="http://ceur-ws.org/Vol-2624/paper5.pdf">http://ceur-ws.org/Vol-2624/paper5.pdf</a>.</p>

</div>

<div id="ref-foth_umfassende_2014">

<p>Foth, Kilian A. 2014. <em>Eine Umfassende Constraint-Dependenz-Grammatik Des Deutschen</em>. Universität Hamburg. <a href="https://edoc.sub.uni-hamburg.de/informatik/volltexte/2014/204/">https://edoc.sub.uni-hamburg.de/informatik/volltexte/2014/204/</a>.</p>

</div>

<div id="ref-krug_description_2017">

<p>Krug, Markus, Lukas Weimer, Isabella Reger, Luisa Macharowsky, Stephan Feldhaus, Frank Puppe, and Fotis Jannidis. 2017. “Description of a Corpus of Character References in German Novels - DROC [Deutsches ROman Corpus].” <a href="https://resolver.sub.uni-goettingen.de/purl?gro-2/108301">https://resolver.sub.uni-goettingen.de/purl?gro-2/108301</a>.</p>

</div>

<div id="ref-kurfali_breaking_2021">

<p>Kurfalı, Murathan, and Mats Wirén. 2021. “Breaking the Narrative: Scene Segmentation Through Sequential Sentence Classification.” In <em>Proceedings of the Shared Task on Scene Segmentation</em>, edited by Albin Zehe, Leonard Konle, Lea Dümpelmann, Evelyn Gius, Svenja Guhr, Andreas Hotho, Fotis Jannidis, et al., 3001:49–53. CEUR Workshop Proceedings. Düsseldorf, Germany. <a href="http://ceur-ws.org/Vol-3001/#paper6">http://ceur-ws.org/Vol-3001/#paper6</a>.</p>

</div>

<div id="ref-proisl_someweta_2018">

<p>Proisl, Thomas. 2018. “SoMeWeTa: A Part-of-Speech Tagger for German Social Media and Web Texts.” In <em>Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)</em>, 665–70. Miyazaki, Japan: European Language Resources Association ELRA. <a href="http://www.lrec-conf.org/proceedings/lrec2018/pdf/49.pdf">http://www.lrec-conf.org/proceedings/lrec2018/pdf/49.pdf</a>.</p>

</div>

<div id="ref-proisl_somajo_2016">

<p>Proisl, Thomas, and Peter Uhrig. 2016. “SoMaJo: State-of-the-Art Tokenization for German Web and Social Media Texts.” In <em>Proceedings of the 10th Web as Corpus Workshop (WAC-X) and the EmpiriST Shared Task</em>, 57–62. Berlin, Germany: Association for Computational Linguistics (ACL). <a href="http://aclweb.org/anthology/W16-2607">http://aclweb.org/anthology/W16-2607</a>.</p>

</div>

<div id="ref-schmid_deep_2019">

<p>———. 2019. “Deep Learning-Based Morphological Taggers and Lemmatizers for Annotating Historical Texts.” In <em>DATeCH, Proceedings of the 3rd International Conference on Digital Access to Textual Cultural Heritage</em>, 133–37. Brussels, Belgium: Association for Computing Machinery. <a href="https://www.cis.uni-muenchen.de/~schmid/papers/Datech2019.pdf">https://www.cis.uni-muenchen.de/~schmid/papers/Datech2019.pdf</a>.</p>

</div>

<div id="ref-schroder_neural_2021">

<p>Schröder, Fynn, Hans Ole Hatzel, and Chris Biemann. 2021. “Neural End-to-End Coreference Resolution for German in Different Domains.” In <em>Proceedings of the 17th Conference on Natural Language Processing (KONVENS 2021)</em>, 170–81. Düsseldorf, Germany: KONVENS 2021 Organizers. <a href="https://aclanthology.org/2021.konvens-1.15">https://aclanthology.org/2021.konvens-1.15</a>.</p>

</div>

<div id="ref-schweter_flert_2021">

<p>Schweter, Stefan, and Alan Akbik. 2021. “FLERT: Document-Level Features for Named Entity Recognition.” <em>arXiv:2011.06993 [Cs]</em>, May. <a href="http://arxiv.org/abs/2011.06993">http://arxiv.org/abs/2011.06993</a>.</p>

</div>

<div id="ref-sennrich_zmorge_2014">

<p>Sennrich, Rico, and Beat Kunz. 2014. “Zmorge: A German Morphological Lexicon Extracted from Wiktionary.” In <em>Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14)</em>, 1063–7. Reykjavik, Iceland: European Language Resources Association (ELRA). <a href="http://www.lrec-conf.org/proceedings/lrec2014/pdf/116_Paper.pdf">http://www.lrec-conf.org/proceedings/lrec2014/pdf/116_Paper.pdf</a>.</p>

</div>

<div id="ref-sennrich_new_2009">

<p>Sennrich, Rico, G. Schneider, M. Volk, M. Warin, C. Chiarcos, Richard Eckart de Castilho, and Manfred Stede. 2009. “A New Hybrid Dependency Parser for German.” In <em>Proceedings of the GSCL Conference</em>. Potsdam, Germany. <a href="https://doi.org/10.5167/UZH-25506">https://doi.org/10.5167/UZH-25506</a>.</p>

</div>

<div id="ref-sennrich_exploiting_2013">

<p>Sennrich, Rico, Martin Volk, and Gerold Schneider. 2013. “Exploiting Synergies Between Open Resources for German Dependency Parsing, POS-Tagging, and Morphological Analysis.” In <em>Proceedings of the International Conference Recent Advances in Natural Language Processing RANLP 2013</em>, 601–9. Hissar, Bulgaria: INCOMA Ltd. Shoumen, BULGARIA. <a href="https://www.aclweb.org/anthology/R13-1079">https://www.aclweb.org/anthology/R13-1079</a>.</p>

</div>

<div id="ref-vauth_automated_2021">

<p>Vauth, Michael, Hans Ole Hatzel, Evelyn Gius, and Chris Biemann. 2021. “Automated Event Annotation in Literary Texts.” In <em>Proceedings of the Conference on Computational Humanities Research 2021</em>, edited by Maud Ehrmann, Folgert Karsdorp, Melvin Wevers, Tara Lee Andrews, Manuel Burghardt, Mike Kestemont, Enrique Manjavacas, Michael Piotrowski, and Joris van Zundert, 2989:333–45. CEUR Workshop Proceedings. Amsterdam, the Netherlands. <a href="https://ceur-ws.org/Vol-2989/#short_paper18">https://ceur-ws.org/Vol-2989/#short_paper18</a>.</p>

</div>
