# Literary Language Processing Pipeline (LLP-Pipeline)

A modular NLP Pipeline for German literary texts. Work in progress.

This pipeline currently performs
* Tokenization and Sentence Splitting via [NLTK](https://www.nltk.org/_modules/nltk/tokenize/punkt.html) [(Bird, Klein, Loper 2009)](#ref-bird_natural_2009)
* POS tagging via [SoMeWeTa](https://github.com/tsproisl/SoMeWeTa) [(Proisl 2018)](#ref-proisl_someweta_2018)
* Lemmatization and Morphological Analysis via [RNNTagger](https://www.cis.uni-muenchen.de/~schmid/tools/RNNTagger/) [(Schmid 2019)](#ref-schmid_deep_2019)
* Dependency Parsing via [ParZu](https://github.com/rsennrich/ParZu) ([Sennrich, Schneider, Volk, Warin 2009](#ref-sennrich_new_2009); [Sennrich, Volk, Schneider 2013](#ref-sennrich_exploiting_2013))

See [Model Selection](./doc/MODEL_SELECTION.md) for a discussion on this choice of language models.

Modules open to implement are:
* Named Entity Recognition via Flair embeddings
* Semantic Role Labeling via InVeRo-XL [(Conia, Orlando, Cecconi, Navigli 2021)](#ref-conia_invero-xl-2021)
* Coreference Resolution via BERT Embeddings [(Schröder, Hatzel, Biemann 2021)](#ref-schroder_neural_2021)
* Tagging of German speech, thought and writing representation (STWR) via Flair/BERT embeddings [(Brunner, Tu, Weimer, Jannidis 2020)](#ref-brunner_bert_2021)

## Prerequisites

* Python 3.7
* For RNNTagger
  * CUDA (tested on version 11.4)
* For Parzu:
  * SWI-Prolog 5.6
  * SFST

## Preparation

Execute `./prepare.sh`, or perform following commands:

```shell
pip install -r requirements.txt
python -c 'import nltk; nltk.download("punkt")

cd resources
wget 'https://corpora.linguistik.uni-erlangen.de/someweta/german_newspaper_2020-05-28.model'
wget 'https://pub.cl.uzh.ch/users/sennrich/zmorge/transducers/zmorge-20150315-smor_newlemma.ca.zip'
unzip zmorge-20150315-smor_newlemma.ca.zip
wget 'https://www.cis.uni-muenchen.de/~schmid/tools/RNNTagger/data/RNNTagger.zip'
unzip -uo RNNTagger.zip
```

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

## Docker Module

A docker module can be built after preparing the installation:

```shell
./prepare.sh && docker build --tag cophiwue/llpipeline
```

Example usage:

```shell
mkdir -p files/in files/out
# copy files into ./files/in to be processed
docker docker run --interactive \
    --tty \
    -a stderr \
    -v "./files:/files" \
    cophiwue/llpipeline -v --writefiles /files/out /files/in
# processed files are located in ./files/out
```

## Developer Guide

See the separate [Developer Guide](./doc/DEVELOPING.md)

## References

<div id="ref-bird_natural_2009" class="csl-entry" role="doc-biblioentry">

Bird, Steven, Ewan Klein, and Edward Loper. 2009. <em>Natural Language Processing with Python</em>. Cambridge, Mass.: O’Reilly.

</div>

<div id="ref-brunner_bert_2021" class="csl-entry" role="doc-biblioentry">

Brunner, Annelen, Ngoc Duyen Tanja Tu, Lukas Weimer, and Fotis Jannidis. 2021. <span>“To <span>BERT</span> or Not to <span>BERT</span> – Comparing Contextual Embeddings in a Deep Learning Architecture for the Automatic Recognition of Four Types of Speech, Thought and Writing Representation.”</span> In <em>Proceedings of the 5th Swiss Text Analytics Conference (<span>SwissText</span>) &amp; 16th Conference on Natural Language Processing (<span>KONVENS</span>)</em>, 2624:11. <span>CEUR</span> Workshop Proceedings. Zurich, Switzerland.

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

<div id="ref-sennrich_new_2009" class="csl-entry" role="doc-biblioentry">

Sennrich, Rico, G. Schneider, M. Volk, M. Warin, C. Chiarcos, Richard Eckart de Castilho, and Manfred Stede. 2009. <span>“A New Hybrid Dependency Parser for German.”</span> In <em>Proceedings of the <span>GSCL</span> Conference</em>. Potsdam, Germany. <a href="https://doi.org/10.5167/UZH-25506">https://doi.org/10.5167/UZH-25506</a>.

</div>

<div id="ref-sennrich_exploiting_2013" class="csl-entry" role="doc-biblioentry">

Sennrich, Rico, Martin Volk, and Gerold Schneider. 2013. <span>“Exploiting Synergies Between Open Resources for German Dependency Parsing, <span>POS</span>-Tagging, and Morphological Analysis.”</span> In <em>Proceedings of the International Conference Recent Advances in Natural Language Processing <span>RANLP</span> 2013</em>, 601–9. Hissar, Bulgaria: <span>INCOMA</span> Ltd. Shoumen, <span>BULGARIA</span>. <a href="https://www.aclweb.org/anthology/R13-1079">https://www.aclweb.org/anthology/R13-1079</a>.

</div>
