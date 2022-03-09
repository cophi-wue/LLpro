#!/bin/sh

# this script has been modified to just do
#  lemmatization
#  (original in `/opt/RNNTagger/cmd/rnn-tagger-german.sh`)

cd ./RNNTagger
SCRIPTS=./scripts
LIB=./lib
PyNMT=./PyNMT
TMP=/tmp/rnn-tagger$$
LANGUAGE=german

REFORMAT=${SCRIPTS}/reformat.pl
LEMMATIZER=$PyNMT/nmt-translate.py
NMTPAR=${LIB}/PyNMT/${LANGUAGE}

$REFORMAT $1 > $TMP.reformatted
python $LEMMATIZER --print_source $NMTPAR $TMP.reformatted > $TMP.lemmas
$SCRIPTS/lemma-lookup.pl $TMP.lemmas $1

rm $TMP.reformatted $TMP.lemmas
