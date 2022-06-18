# Output Formats

## Raw Pipeline Output

The default implementation of the LLP-Pipeline writes one output file for each input file.

Each line in the output file corresponds to a JSON object, which contains the annotations of the fields and metadata.

* `obj[field][module].value` holds the annotation value for `field` written by `module`, i.e. `tok.get_field(field, module)`.
* `obj[field][module].metadata` holds the annotation value for `metadata` written by `module`, i.e. `tok.get_metadata(field, module)`.

Use, e.g., `tokens = [Token.read_json(line) for line in raw_output_file]` to retrieve the list of processed tokens from an output file.

## Conversion to Tabular Output

To facilitate IO of the output files, a conversion script is provided that converts the raw JSON output to a TSV file, containing a flat table of the JSON data (requires only `pandas`).

```
python ./format_tsv.py raw_output_file tsv_file
```

For example, the column names of the tabular output converted from raw output of the default pipeline look like this:

* `pos.RNNTagger`, holds the value of the field `pos` from module `RNNTagger`.
* `pos.RNNTagger.meta.prob`, holds the metadata value `prob` of the field `pos` from module `SoMaJoTokenizer`.
* ...

Use, e.g., `Token.to_dataframe(Token.read_json(line) for line in raw_output_file)` to retrieve the processed tokens as Pandas DataFrame from a raw output file.

# Annotation Format of the Modules, Used Tagsets

* Tokenizations results from SoMaJoTokenizer:

  * Writes to fields `token`, `id`, `sentence`, `doc`, corresponding to token string, token id, sentence id, document filename.

  * Metadata dict containing `space_after` and `token_class` values returned by the SoMaJo tokenizer.
  

* POS Tags from SoMeWeTaTagger:
 
  * Uses [TIGER variant of the STTS tagset](https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/tiger-corpus/annotation/tiger_scheme-syntax.pdf) (field `pos`). [Overview](https://www.linguistik.hu-berlin.de/de/institut/professuren/korpuslinguistik/mitarbeiter-innen/hagen/STTS_Tagset_Tiger).
    Additonally, provides automatic conversion to [Universal Dependencies v2 POS tags](https://universaldependencies.org/u/pos/all.html) (field `upos`).
    Conversion performed using table [*de::stts*](https://universaldependencies.org/tagset-conversion/de-stts-uposf.html).
  
 
* Morphological Analysis from RNNTagger:

  * Writes to field `morph` the corresponding morphological analysis. (Additionally, writes to `pos` and `upos` fields the POS results of the RNNTagger).

  * Uses the [TIGER Morphology Annotation Scheme](https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/tiger-corpus/annotation/tiger_scheme-morph.pdf).

  * Metadata dict holds in key `prob` the softmax-normalized score for the particular morphological analysis.
  

* Lemmatization from RNNLemmatizer:

  * Writes to field `lemma` the result of the lemmatization.

  * Metadata dict holds in key `prob` the softmax-normalized score for the particular lemmatization.
  
  
* Dependency relation labels from ParzuParser:

  * Writes to field `deprel` the corresponding dependency relation type, and to field `head` the `id` of the corresponding head token in the same sentence.

  * Uses “Eine umfassende Constraint-Dependenz-Grammatik des Deutschen” ([Forth, 2005](#ref-forth_umfassende_2014)) as labels, used in, e.g., the Hamburg Dependency Treebank. [Overview](https://github.com/rsennrich/ParZu/blob/master/doc/LABELS.md).
  
 
* Named Entites from FLERTNERTagger:

  * Writes to field `ner` an array of NER lables (possibly empty).
 
  * Uses the usual four classes `PER, LOC, ORG, MISC` as employed by the respective CoNLL-2003 shared task on named entity recognition. ([Sang and De Meulder, 2003](#ref-tjong_kim_sang_introduction_2003))

  * Metadata dict holds in key `prob` the softmax-normalized scores for each of the annotation values in the array held by field `ner`. Holds in key `ids` an array of (document-unique) IDs corresponding the annotated named entities. The tokens compromising the same entity contain the same ID in that field.
  

* RedewiedergabeTagger:

  * Writes to field `speech_direct`, `speech_indirect`, `speech_reported`, `speech_freeIndirect` one of the binary annotation values `no` or `yes`
  * Metadata dict holds in key `prob` the normalized probability for the particular annotation value.
  

* Coreference Resolution from CorefIncrementalTagger:

  * Writes to `coref_clusters` a list of document-unique IDs that correspond to mentions. The tokens compromising the same mention contain the same ID in that field.


* Semantic Role Labels from InVeRoXL:

  * Writes to field `srl` a (possibly empty) list of *frames*, where each frame is a dict.
    Each dict holds in key `id` a document-unique ID of the particular semantic frame. 
    At the token corresponding to the verb of the frame, the frame dict holds in key `sense` the frame name. At the tokens corresponding to roles of the frame, the frame dict holds in key `role` the type of role.

  * Frames and Semantic Roles defined by VerbAtlas ([Di Fabio et al., 2019](#ref-di_fabio_verbatlas_2019)). [List of Semantic Roles](https://verbatlas.org/semantic), [List of Frames](https://verbatlas.org/frames).


# References

<div id="ref-di_fabio_verbatlas_2019" class="csl-entry" role="doc-biblioentry">

Di Fabio, Andrea, Simone Conia, and Roberto Navigli. 2019. <span>“<span>VerbAtlas</span>: A Novel Large-Scale Verbal Semantic Resource and Its Application to Semantic Role Labeling.”</span> In <em>Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (<span>EMNLP</span>-<span>IJCNLP</span>)</em>, 627–37. Hong Kong, China: Association for Computational Linguistics. <a href="https://doi.org/10.18653/v1/D19-1058">https://doi.org/10.18653/v1/D19-1058</a>.

</div>

<div id="ref-forth_umfassende_2014" class="csl-entry" role="doc-biblioentry">

Forth, Kilian A. 2014. <em>Eine Umfassende Constraint-Dependenz-Grammatik Des Deutschen</em>. Universität Hamburg. <a href="https://edoc.sub.uni-hamburg.de/informatik/volltexte/2014/204/">https://edoc.sub.uni-hamburg.de/informatik/volltexte/2014/204/</a>.

</div>

<div id="ref-tjong_kim_sang_introduction_2003" class="csl-entry" role="doc-biblioentry">

Sang, Erik F. Tjong Kim, and Fien De Meulder. 2003. <span>“Introduction to the <span>CoNLL</span>-2003 Shared Task: Language-Independent Named Entity Recognition.”</span> In <em>Proceedings of the Seventh Conference on Natural Language Learning at <span>HLT</span>-<span>NAACL</span> 2003</em>, 142–47. Edmonton, Canada: Association for Computational Linguistics. <a href="https://doi.org/10.3115/1119176.1119195">https://doi.org/10.3115/1119176.1119195</a>.

</div>
