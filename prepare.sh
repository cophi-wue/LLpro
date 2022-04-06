#!/usr/bin/bash

pip install -r requirements.txt
python -c 'import nltk; nltk.download("punkt")'

cd resources
wget 'https://corpora.linguistik.uni-erlangen.de/someweta/german_newspaper_2020-05-28.model'
wget 'https://pub.cl.uzh.ch/users/sennrich/zmorge/transducers/zmorge-20150315-smor_newlemma.ca.zip'
unzip zmorge-20150315-smor_newlemma.ca.zip
wget 'https://www.cis.uni-muenchen.de/~schmid/tools/RNNTagger/data/RNNTagger.zip'
unzip -uo RNNTagger.zip
find ./RNNTagger/lib/ -type f ! -name '*german*' -delete # remove unncessesary models

