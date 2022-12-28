# LLpro – A Literary Language Processing Pipeline for German Narrative Texts

A modular NLP Pipeline for German literary texts implemented in Python and Spacy. Work in progress.

This pipeline currently performs
* Tokenization and Sentence Splitting via [SoMaJo](https://github.com/tsproisl/SoMaJo)  [(Proisl, Uhrig 2016)](#ref-proisl_somajo_2016).
* POS tagging via [SoMeWeTa](https://github.com/tsproisl/SoMeWeTa) [(Proisl 2018)](#ref-proisl_someweta_2018)
* Lemmatization and Morphological Analysis via [RNNTagger](https://www.cis.uni-muenchen.de/~schmid/tools/RNNTagger/) [(Schmid 2019)](#ref-schmid_deep_2019)
* Dependency Parsing via [ParZu](https://github.com/rsennrich/ParZu) ([Sennrich, Schneider, Volk, Warin 2009](#ref-sennrich_new_2009); [Sennrich, Volk, Schneider 2013](#ref-sennrich_exploiting_2013); [Sennrich, Kunz 2014](#ref-sennrich_zmorge_2014))
* Named Entity Recognition via [FLERT](https://github.com/flairNLP/flair) [(Schweter, Akbik 2021)](#ref-schweter_flert_2021)
* Coreference Resolution via BERT Embeddings [(Schröder, Hatzel, Biemann 2021)](#ref-schroder_neural_2021)
* Tagging of German speech, thought and writing representation (STWR) via Flair/BERT embeddings [(Brunner, Tu, Weimer, Jannidis 2020)](#ref-brunner_bert_2021)
* Segmentation into Scenes via BERT Embeddings [(Kurfalı and Wirén 2021)](#ref-kurfali_breaking_2021)
* Annotating Event Types to verbal phrases via BERT Embeddings [(Vauth, Hatzel, Gius, Biemann 2021)](#ref-vauth_automated_2021)

See [Model Selection](./doc/MODEL_SELECTION.md) for a discussion on this choice of language models.

See also the section about the [Output Format](./doc/OUTPUT_FORMAT.md) for a description of the tabular output format.

## Usage

```text
usage: main.py [-h] [-v] [--paragraph-pattern PAT] [--section-pattern PAT]
               [--stdout | --writefiles DIR]
               FILE [FILE ...]

NLP Pipeline for literary texts written in German.

required arguments:
  --infiles FILE [FILE ...]
                        Input files, or directories.

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose
  --paragraph-pattern PAT
                        Optional paragraph separator pattern. Paragraph
                        separators are removed, and sentences always terminate
                        on paragraph boundaries.
  --section-pattern PAT
                        Optional sectioning paragraph pattern. Paragraphs
                        fully matching the pattern are removed, and increment
                        the section id counter for tokens in intermediate
                        paragraphs.
  --stdout              Write all processed tokens to stdout.
  --writefiles DIR      For each input file, write processed tokens to a
                        separate file in DIR.
```

## Installation

The LLpro pipeline can be run either locally or as a Docker container. Running
the pipelie using Docker is strongly recommended.


**WINDOWS USERS**: For building the Docker image, clone using
```shell
git clone https://github.com/aehrm/LLpro --config core.autocrlf=input
```
to preserve line endings.

### Building and running the Docker image

We strongly recommend using Docker to run the pipeline. With the provided
Dockerfile, all  dependencies and prerequisites are downloaded
automatically.

```shell
docker build --tag cophiwue/llpro .
```
Optionally, NVIDIA Apex can be installed inside the Docker image for faster GPU inference.
```shell
docker build --build-arg=INSTALL_APEX=1 --tag cophiwue/llpro .
```

After building, the Docker image can be run like this:

```shell
mkdir -p files/in files/out
# copy files into ./files/in to be processed
docker run \
    --rm \
    --cpus 4 \ 
    --gpus all \    # alternatively, e.g., --gpus '"device=0"'
    --interactive \
    --tty \
    -a stdout \
    -a stderr \
    -v "$(pwd)/files:/files" \
    cophiwue/llpro -v --writefiles /files/out --infiles /files/in
# processed files are located in ./files/out
```

### Installing locally

Verify that the following dependencies are installed:

* Python (tested on version 3.7)
* For RNNTagger
  * CUDA (tested on version 11.4)
* For Parzu:
  * SWI-Prolog >= 5.6
  * SFST >= 1.4

Execute `pip install -r requirements.txt` and `./prepare.sh`. The script downloads all remaining prerequisites.
Execute `./setup.py build_ext --inplace` to complie the required Cython files.

Example usage:

```shell
pip install -r requirements.txt
./prepare.sh
./setup.py build_ext --inplace
python ./main.py -v --writefiles files/out files/in
```


## Developer Guide

See the separate [Developer Guide](./doc/DEVELOPING.md) about the implemented Spacy components and how to access the assigned attributes.

See also the separate document about the tabular [Output Format](./doc/OUTPUT_FORMAT.md) for a description of the output format and a reference of the used tagsets.

## Citing

If you use the LLpro software for academic research, please consider citing the project:

Ehrmanntraut, Anton, Leonard Konle, and Fotis Jannidis: *LLpro*, A Literary Language Processing Pipeline for German
Narrative Texts. 2022. <https://github.com/aehrm/LLpro>

## License

In accordance with the license terms of ParZu+Zmorge (GPL v2), and of SoMeWeTa
(GPL v3) the LLpro pipeline is licensed under the terms of GPL v3. See
[LICENSE](LICENSE.md). NOTICE: Some subsystems and resources used by the
LLpro pipeline have additional license terms:

* RNNTagger: see
  <https://www.cis.uni-muenchen.de/~schmid/tools/RNNTagger/Tagger-Licence>
* SoMeWeTa model `german_web_social_media_2020-05-28.model`: derived from the
  TIGER corpus; see
<https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/tiger-corpus/license/htmlicense.html>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

## References

<div id="ref-akbik_contextual_2018">

<p>Akbik, Alan, Duncan Blythe, and Roland Vollgraf. 2018. “Contextual String Embeddings for Sequence Labeling.” In <em>COLING 2018, 27th International Conference on Computational Linguistics</em>, 1638–49.</p>

</div>

<div id="ref-altinok_demorphy_2018">

<p>Altinok, Duygu. 2018. “DEMorphy, German Language Morphological Analyzer.” <a href="https://arxiv.org/abs/1803.00902">https://arxiv.org/abs/1803.00902</a>.</p>

</div>

<div id="ref-bird_natural_2009">

<p>Bird, Steven, Ewan Klein, and Edward Loper. 2009. <em>Natural Language Processing with Python</em>. Cambridge, Mass.: O’Reilly.</p>

</div>

<div id="ref-brunner_bert_2021">

<p>Brunner, Annelen, Ngoc Duyen Tanja Tu, Lukas Weimer, and Fotis Jannidis. 2021. “To BERT or Not to BERT – Comparing Contextual Embeddings in a Deep Learning Architecture for the Automatic Recognition of Four Types of Speech, Thought and Writing Representation.” In <em>Proceedings of the 5th Swiss Text Analytics Conference (SwissText) &amp; 16th Conference on Natural Language Processing (KONVENS)</em>, 2624:11. CEUR Workshop Proceedings. Zurich, Switzerland. <a href="http://ceur-ws.org/Vol-2624/paper5.pdf">http://ceur-ws.org/Vol-2624/paper5.pdf</a>.</p>

</div>

<div id="ref-conia_invero-xl_2021">

<p>Conia, Simone, Riccardo Orlando, Fabrizio Brignone, Francesco Cecconi, and Roberto Navigli. 2021. “InVeRo-XL: Making Cross-Lingual Semantic Role Labeling Accessible with Intelligible Verbs and Roles.” In <em>Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations</em>, 319–28. Online; Punta Cana, Dominican Republic: Association for Computational Linguistics. <a href="https://doi.org/10.18653/v1/2021.emnlp-demo.36">https://doi.org/10.18653/v1/2021.emnlp-demo.36</a>.</p>

</div>

<div id="ref-di_fabio_verbatlas_2019">

<p>Di Fabio, Andrea, Simone Conia, and Roberto Navigli. 2019. “VerbAtlas: A Novel Large-Scale Verbal Semantic Resource and Its Application to Semantic Role Labeling.” In <em>Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)</em>, 627–37. Hong Kong, China: Association for Computational Linguistics. <a href="https://doi.org/10.18653/v1/D19-1058">https://doi.org/10.18653/v1/D19-1058</a>.</p>

</div>

<div id="ref-forth_umfassende_2014">

<p>Forth, Kilian A. 2014. <em>Eine Umfassende Constraint-Dependenz-Grammatik Des Deutschen</em>. Universität Hamburg. <a href="https://edoc.sub.uni-hamburg.de/informatik/volltexte/2014/204/">https://edoc.sub.uni-hamburg.de/informatik/volltexte/2014/204/</a>.</p>

</div>

<div id="ref-kiss_unsupervised_2006">

<p>Kiss, Tibor, and Jan Strunk. 2006. “Unsupervised Multilingual Sentence Boundary Detection.” <em>Computational Linguistics</em> 32 (4): 485–525. <a href="https://doi.org/10.1162/coli.2006.32.4.485">https://doi.org/10.1162/coli.2006.32.4.485</a>.</p>

</div>

<div id="ref-kurfali_breaking_2021">

<p>Kurfalı, Murathan, and Mats Wirén. 2021. “Breaking the Narrative: Scene Segmentation Through Sequential Sentence Classification.” In <em>Proceedings of the Shared Task on Scene Segmentation</em>, edited by Albin Zehe, Leonard Konle, Lea Dümpelmann, Evelyn Gius, Svenja Guhr, Andreas Hotho, Fotis Jannidis, et al., 3001:49–53. CEUR Workshop Proceedings. Düsseldorf, Germany. <a href="http://ceur-ws.org/Vol-3001/#paper6">http://ceur-ws.org/Vol-3001/#paper6</a>.</p>

</div>

<div id="ref-lavergne_practical_2010">

<p>Lavergne, Thomas, Olivier Cappé, and François Yvon. 2010. “Practical Very Large Scale CRFs.” In <em>Proceedings the 48th Annual Meeting of the Association for Computational Linguistics (ACL)</em>, 504–13. Association for Computational Linguistics. <a href="http://www.aclweb.org/anthology/P10-1052">http://www.aclweb.org/anthology/P10-1052</a>.</p>

</div>

<div id="ref-manning_stanford_2014">

<p>Manning, Christopher D., Mihai Surdeanu, John Bauer, Jenny Finkel, Steven J. Bethard, and David McClosky. 2014. “The Stanford CoreNLP Natural Language Processing Toolkit.” In <em>Proceedings of 52nd Annual Meeting of the Association for Computational Linguistics: System Demonstrations</em>, 55–60. Association for Computational Linguistics. <a href="http://www.aclweb.org/anthology/P/P14/P14-5010">http://www.aclweb.org/anthology/P/P14/P14-5010</a>.</p>

</div>

<div id="ref-mueller_efficient_2013">

<p>Mueller, Thomas, Helmut Schmid, and Hinrich Schütze. 2013. “Efficient Higher-Order CRFs for Morphological Tagging.” In <em>Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing</em>, 322–32. Seattle, Washington, USA: Association for Computational Linguistics. <a href="https://www.aclweb.org/anthology/D13-1032">https://www.aclweb.org/anthology/D13-1032</a>.</p>

</div>

<div id="ref-ortmann_evaluating_2019">

<p>Ortmann, Katrin, A. Roussel, and Stefanie Dipper. 2019. “Evaluating Off-the-Shelf NLP Tools for German.” In <em>Proceedings of the 15th Conference on Natural Language Processing (KONVENS 2019)</em>, 212–22. Erlangen, Germany: German Society for Computational Linguistics &amp; Language Technology. <a href="https://konvens.org/proceedings/2019/papers/KONVENS2019_paper_55.pdf">https://konvens.org/proceedings/2019/papers/KONVENS2019_paper_55.pdf</a>.</p>

</div>

<div id="ref-proisl_someweta_2018">

<p>Proisl, Thomas. 2018. “SoMeWeTa: A Part-of-Speech Tagger for German Social Media and Web Texts.” In <em>Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)</em>, 665–70. Miyazaki, Japan: European Language Resources Association ELRA. <a href="http://www.lrec-conf.org/proceedings/lrec2018/pdf/49.pdf">http://www.lrec-conf.org/proceedings/lrec2018/pdf/49.pdf</a>.</p>

</div>

<div id="ref-proisl_somajo_2016">

<p>Proisl, Thomas, and Peter Uhrig. 2016. “SoMaJo: State-of-the-Art Tokenization for German Web and Social Media Texts.” In <em>Proceedings of the 10th Web as Corpus Workshop (WAC-X) and the EmpiriST Shared Task</em>, 57–62. Berlin, Germany: Association for Computational Linguistics (ACL). <a href="http://aclweb.org/anthology/W16-2607">http://aclweb.org/anthology/W16-2607</a>.</p>

</div>

<div id="ref-tjong_kim_sang_introduction_2003">

<p>Sang, Erik F. Tjong Kim, and Fien De Meulder. 2003. “Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition.” In <em>Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003</em>, 142–47. Edmonton, Canada: Association for Computational Linguistics. <a href="https://doi.org/10.3115/1119176.1119195">https://doi.org/10.3115/1119176.1119195</a>.</p>

</div>

<div id="ref-schmid_probabilistic_1994">

<p>Schmid, Helmut. 1994. “Probabilistic Part-of-Speech Tagging Using Decision Trees.” In <em>Proceedings of International Conference on New Methods in Language Processing</em>. Manchester, UK. <a href="https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger1.pdf">https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger1.pdf</a>.</p>

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
