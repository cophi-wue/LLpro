# Literary Language Processing Pipeline (LLP-Pipeline)

A modular NLP Pipeline for German literary texts.

This pipeline currently performs
* Tokenization via NLTK (Kiss and Strunk 2006)
* POS tagging via SoMeWeTa (Proisl 2018)
* Lemmatization and Morphological Analysis via RNNTagger (Schmid 2019)
* Dependency Parsing via ParZu (Sennrich, Volk, Schneider 2013)

See [Model Selection](./doc/MODEL_SELECTION.md) for a discussion on this choice of language models.

Modules open to implement are:
* Named Entity Recognition via Flair embeddings
* Semantic Role Labeling via InVeRo-XL (Conia et al.)
* Coreference Resolution via BERT Embeddings (Schröder et al.)
* Tagging of German speech, thought and writing representation (STWR) via Flair/BERT embeddings (Brunner, Tu, Weimer, Jannidis 2020)

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
docker docker run --interactive \
    --tty \
    -a stderr \
    -v "./files:/files" \
    llptest -v --writefiles /files/out /files/in
```

## Developer Guide

See the separate [Developer Guide](./doc/DEVELOPING.md)


