# LLpro – A Literary Language Processing Pipeline for German Narrative Texts

An NLP Pipeline for German literary texts implemented in Python and Spacy (v3.3.1). Work in progress.

This pipeline implements several custom pipeline components using the Spacy API. Currently the components perform
* Tokenization and Sentence Splitting via [SoMaJo](https://github.com/tsproisl/SoMaJo)  [(Proisl, Uhrig 2016)](#ref-proisl_somajo_2016). Version 2.2.
* POS tagging via [SoMeWeTa](https://github.com/tsproisl/SoMeWeTa) [(Proisl 2018)](#ref-proisl_someweta_2018). Version 1.8.1.
* Lemmatization and Morphological Analysis via [RNNTagger](https://www.cis.uni-muenchen.de/~schmid/tools/RNNTagger/) [(Schmid 2019)](#ref-schmid_deep_2019). Version 1.4.1.
* Dependency Parsing via [ParZu](https://github.com/rsennrich/ParZu) ([Sennrich, Schneider, Volk, Warin 2009](#ref-sennrich_new_2009); [Sennrich, Volk, Schneider 2013](#ref-sennrich_exploiting_2013); [Sennrich, Kunz 2014](#ref-sennrich_zmorge_2014)). Commit bdc466d.
* Named Entity Recognition via [FLERT](https://github.com/flairNLP/flair) [(Schweter, Akbik 2021)](#ref-schweter_flert_2021). Commit 2f017e.
* Coreference Resolution via BERT Embeddings [(Schröder, Hatzel, Biemann 2021)](#ref-schroder_neural_2021). Commit f34a99e.
* Tagging of German speech, thought and writing representation (STWR) via Flair/BERT embeddings [(Brunner, Tu, Weimer, Jannidis 2020)](#ref-brunner_bert_2021). Version 1.0.0.
* Segmentation into Scenes via BERT Embeddings [(Kurfalı and Wirén 2021)](#ref-kurfali_breaking_2021) Commit 656c37b.
* Annotating Event Types to verbal phrases via BERT Embeddings [(Vauth, Hatzel, Gius, Biemann 2021)](#ref-vauth_automated_2021) Version 0.2, Commit 25fdf7e.

See [Model Selection](./doc/MODEL_SELECTION.md) for a discussion on this choice of language models.

See also the section about the [Output Format](./doc/OUTPUT_FORMAT.md) for a description of the tabular output format.

## Usage

```text
usage: bin/llpro_cli.py [-h] [-v] [--paragraph-pattern PAT] [--section-pattern PAT]
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

Note: you can specify the resources directory (containing `ParZu` etc.) with the environment
variable `LLPRO_RESOURCES_ROOT`, and the temporary workdir with the environment variable `LLPRO_TEMPDIR`.

## Installation

The LLpro pipeline can be run either locally or as a Docker container. Running
the pipeline using Docker is strongly recommended.


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

Execute `poetry install` and `./prepare.sh`. The script downloads all remaining prerequisites.
Example usage:

```shell
poetry install
./prepare.sh
# NOTICE: use the prepared poetry venv!
poetry run python ./bin/llpro_cli.py -v --writefiles files/out files/in

# if desired, run tests
poetry run pytest -vv
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
[LICENSE](LICENSE.md). 

NOTICE: The code of the ParZu parser located in `resources/ParZu` has been modified to be compatible with LLpro.
See `git log -p df1e91a.. -- resources/ParZu` for a summary of these changes.

NOTICE: Some subsystems and resources used by the
LLpro pipeline have additional license terms:

* RNNTagger: see
  <https://www.cis.uni-muenchen.de/~schmid/tools/RNNTagger/Tagger-Licence>
* SoMeWeTa model `german_web_social_media_2020-05-28.model`: derived from the
  TIGER corpus; see
<https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/tiger-corpus/license/htmlicense.html>
* Scene Segmenter by Kurfalı and Wirén: Code provided as is on Github by the author, but no license terms provided.

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

<div id="ref-brunner_bert_2021">

<p>Brunner, Annelen, Ngoc Duyen Tanja Tu, Lukas Weimer, and Fotis Jannidis. 2021. “To BERT or Not to BERT – Comparing Contextual Embeddings in a Deep Learning Architecture for the Automatic Recognition of Four Types of Speech, Thought and Writing Representation.” In <em>Proceedings of the 5th Swiss Text Analytics Conference (SwissText) &amp; 16th Conference on Natural Language Processing (KONVENS)</em>, 2624:11. CEUR Workshop Proceedings. Zurich, Switzerland. <a href="http://ceur-ws.org/Vol-2624/paper5.pdf">http://ceur-ws.org/Vol-2624/paper5.pdf</a>.</p>

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
