# Literary Language Processing Pipeline (LLP-Pipeline)

A modular NLP Pipeline for German literary texts. Work in progress.

This pipeline currently performs
* Tokenization and Sentence Splitting via [NLTK](https://www.nltk.org/_modules/nltk/tokenize/punkt.html) [(Bird, Klein, Loper 2009)](#ref-bird_natural_2009)
* POS tagging via [SoMeWeTa](https://github.com/tsproisl/SoMeWeTa) [(Proisl 2018)](#ref-proisl_someweta_2018)
* Lemmatization and Morphological Analysis via [RNNTagger](https://www.cis.uni-muenchen.de/~schmid/tools/RNNTagger/) [(Schmid 2019)](#ref-schmid_deep_2019)
* Dependency Parsing via [ParZu](https://github.com/rsennrich/ParZu) ([Sennrich, Schneider, Volk, Warin 2009](#ref-sennrich_new_2009); [Sennrich, Volk, Schneider 2013](#ref-sennrich_exploiting_2013); [Sennrich, Kunz 2014](#ref-sennrich_zmorge_2014))
* Named Entity Recognition via [FLERT](https://github.com/flairNLP/flair) [(Schweter, Akbik 2021)](#ref-schweter_flert_2021)
* Coreference Resolution via BERT Embeddings [(Schröder, Hatzel, Biemann 2021)](#ref-schroder_neural_2021)
* Tagging of German speech, thought and writing representation (STWR) via Flair/BERT embeddings [(Brunner, Tu, Weimer, Jannidis 2020)](#ref-brunner_bert_2021)
* OPTIONAL: Semantic Role Labeling via InVeRo-XL [(Conia, Orlando, Cecconi, Navigli 2021)](#ref-conia_invero-xl-2021)

See [Model Selection](./doc/MODEL_SELECTION.md) for a discussion on this choice of language models.

## Usage

```text
usage: main.py [-h] [-v] [--format {json,conll}]
               [--stdout | --writefiles DIR]
               FILE [FILE ...]

NLP Pipeline for literary texts written in German.

positional arguments:
  FILE

options:
  -h, --help            show this help message and exit
  -v, --verbose
  --format {json,conll}
  --stdout              Write all processed tokens to stdout
  --writefiles DIR      For each input file, write processed
                        tokens to a separate file in DIR
```

## Installation

The LLP-Pipeline can be run either locally or as a Docker container. Running
the pipelie using Docker is strongly recommended.

In both alternatives, Semantic Role Labeling can be enabled by requesting and
downloading a copy of the InVeRo-XL Docker image `invero-xl-span-cuda:2.0.0`
from `http://nlp.uniroma1.it/resources/`. Before continuing, place the extracted Docker image file
`invero-xl-span-cuda-2.0.0.tar` into the folder `resources/`. (Verify that the
tarfile contains the file `manifest.json`.)

### Building and running the Docker image

We strongly recommend using Docker to run the pipeline. With the provided
Dockerfile, all (freely available) dependencies and prerequisites are downloaded
automatically. Before building, place the InVeRo-XL Docker image into the folder `resources/`, if desired.

```shell
docker build --tag cophiwue/llp-pipeline .
```

Example usage:

```shell
mkdir -p files/in files/out
# copy files into ./files/in to be processed
docker run \
    --cpus 4 \
    --runtime nvidia \
    --interactive \
    --tty \
    -a stdout \
    -a stderr \
    -v "$(pwd)/files:/files" \
    cophiwue/llp-pipeline -v --writefiles /files/out /files/in
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
* OPTIONAL: Docker to extract the InVeRo-XL image

Execute `./prepare.sh`. The script downloads all remaining prerequisites.

Example usage:

```shell
python ./main.py -v --writefiles files/out files/in
```


## Developer Guide

See the separate [Developer Guide](./doc/DEVELOPING.md)

## License

In accordance with the license terms of ParZu+Zmorge (GPL v2), and of SoMeWeTa
(GPL v3) the LLP-Pipeline is licensed unter the terms of GPL v3. See
[LICENSE](LICENSE.md). NOTICE: Some subsystems and resources used by the
LLP-Pipeline have additional license terms:

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

<div id="ref-bird_natural_2009" class="csl-entry" role="doc-biblioentry">

Bird, Steven, Ewan Klein, and Edward Loper. 2009. <em>Natural Language Processing with Python</em>. Cambridge, Mass.: O’Reilly.

</div>

<div id="ref-brunner_bert_2021" class="csl-entry" role="doc-biblioentry">

Brunner, Annelen, Ngoc Duyen Tanja Tu, Lukas Weimer, and Fotis Jannidis. 2021. <span>“To <span>BERT</span> or Not to <span>BERT</span> – Comparing Contextual Embeddings in a Deep Learning Architecture for the Automatic Recognition of Four Types of Speech, Thought and Writing Representation.”</span> In <em>Proceedings of the 5th Swiss Text Analytics Conference (<span>SwissText</span>) &amp; 16th Conference on Natural Language Processing (<span>KONVENS</span>)</em>, 2624:11. <span>CEUR</span> Workshop Proceedings. Zurich, Switzerland. <a href="http://ceur-ws.org/Vol-2624/paper5.pdf">http://ceur-ws.org/Vol-2624/paper5.pdf</a>.

</div>

<div id="ref-conia_invero-xl_2021" class="csl-entry" role="doc-biblioentry">

Conia, Simone, Riccardo Orlando, Fabrizio Brignone, Francesco Cecconi, and Roberto Navigli. 2021. <span>“<span>InVeRo</span>-<span>XL</span>: Making Cross-Lingual Semantic Role Labeling Accessible with Intelligible Verbs and Roles.”</span> In <em>Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations</em>, 319–28. Online; Punta Cana, Dominican Republic: Association for Computational Linguistics. <a href="https://doi.org/10.18653/v1/2021.emnlp-demo.36">https://doi.org/10.18653/v1/2021.emnlp-demo.36</a>.

</div>

<div id="ref-ortmann_evaluating_2019" class="csl-entry" role="doc-biblioentry">

Ortmann, Katrin, A. Roussel, and Stefanie Dipper. 2019. <span>“Evaluating Off-the-Shelf <span>NLP</span> Tools for German.”</span> In <em>Proceedings of the 15th Conference on Natural Language Processing (<span>KONVENS</span> 2019)</em>, 212–22. Erlangen, Germany: German Society for Computational Linguistics &amp; Language Technology. <a href="https://konvens.org/proceedings/2019/papers/KONVENS2019_paper_55.pdf">https://konvens.org/proceedings/2019/papers/KONVENS2019_paper_55.pdf</a>.

</div>

<div id="ref-proisl_someweta_2018" class="csl-entry" role="doc-biblioentry">

Proisl, Thomas. 2018. <span>“<span>SoMeWeTa</span>: A Part-of-Speech Tagger for German Social Media and Web Texts.”</span> In <em>Proceedings of the Eleventh International Conference on Language Resources and Evaluation (<span>LREC</span> 2018)</em>, 665–70. Miyazaki, Japan: European Language Resources Association <span>ELRA</span>. <a href="http://www.lrec-conf.org/proceedings/lrec2018/pdf/49.pdf">http://www.lrec-conf.org/proceedings/lrec2018/pdf/49.pdf</a>.

</div>

<div id="ref-schmid_deep_2019" class="csl-entry" role="doc-biblioentry">

Schmid, Helmut. 2019. <span>“Deep Learning-Based Morphological Taggers and Lemmatizers for Annotating Historical Texts.”</span> In <em><span>DATeCH</span>, Proceedings of the 3rd International Conference on Digital Access to Textual Cultural Heritage</em>, 133–37. Brussels, Belgium: Association for Computing Machinery. <a href="https://www.cis.uni-muenchen.de/~schmid/papers/Datech2019.pdf">https://www.cis.uni-muenchen.de/~schmid/papers/Datech2019.pdf</a>.

</div>

<div id="ref-schroder_neural_2021" class="csl-entry" role="doc-biblioentry">

Schröder, Fynn, Hans Ole Hatzel, and Chris Biemann. 2021. <span>“Neural End-to-End Coreference Resolution for German in Different Domains.”</span> In <em>Proceedings of the 17th Conference on Natural Language Processing (<span>KONVENS</span> 2021)</em>, 170–81. Düsseldorf, Germany: <span>KONVENS</span> 2021 Organizers. <a href="https://aclanthology.org/2021.konvens-1.15">https://aclanthology.org/2021.konvens-1.15</a>.

</div>

<div id="ref-schweter_flert_2021" class="csl-entry" role="doc-biblioentry">

Schweter, Stefan, and Alan Akbik. 2021. <span>“<span>FLERT</span>: Document-Level Features for Named Entity Recognition.”</span> <em><span>arXiv</span>:2011.06993 [Cs]</em>, May. <a href="http://arxiv.org/abs/2011.06993">http://arxiv.org/abs/2011.06993</a>.

</div>

<div id="ref-sennrich_zmorge_2014" class="csl-entry" role="doc-biblioentry">

Sennrich, Rico, and Beat Kunz. 2014. <span>“Zmorge: A German Morphological Lexicon Extracted from Wiktionary.”</span> In <em>Proceedings of the Ninth International Conference on Language Resources and Evaluation (<span>LREC</span>’14)</em>, 1063–67. Reykjavik, Iceland: European Language Resources Association (<span>ELRA</span>). <a href="http://www.lrec-conf.org/proceedings/lrec2014/pdf/116_Paper.pdf">http://www.lrec-conf.org/proceedings/lrec2014/pdf/116_Paper.pdf</a>.

</div>

<div id="ref-sennrich_new_2009" class="csl-entry" role="doc-biblioentry">

Sennrich, Rico, G. Schneider, M. Volk, M. Warin, C. Chiarcos, Richard Eckart de Castilho, and Manfred Stede. 2009. <span>“A New Hybrid Dependency Parser for German.”</span> In <em>Proceedings of the <span>GSCL</span> Conference</em>. Potsdam, Germany. <a href="https://doi.org/10.5167/UZH-25506">https://doi.org/10.5167/UZH-25506</a>.

</div>

<div id="ref-sennrich_exploiting_2013" class="csl-entry" role="doc-biblioentry">

Sennrich, Rico, Martin Volk, and Gerold Schneider. 2013. <span>“Exploiting Synergies Between Open Resources for German Dependency Parsing, <span>POS</span>-Tagging, and Morphological Analysis.”</span> In <em>Proceedings of the International Conference Recent Advances in Natural Language Processing <span>RANLP</span> 2013</em>, 601–9. Hissar, Bulgaria: <span>INCOMA</span> Ltd. Shoumen, <span>BULGARIA</span>. <a href="https://www.aclweb.org/anthology/R13-1079">https://www.aclweb.org/anthology/R13-1079</a>.

</div>
