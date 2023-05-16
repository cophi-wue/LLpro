#!/usr/bin/bash
set -e
WGET="wget --no-clobber --show-progress --progress=bar:force:noscroll"

cd resources
$WGET 'https://corpora.linguistik.uni-erlangen.de/someweta/german_newspaper_2020-05-28.model'
$WGET 'https://pub.cl.uzh.ch/users/sennrich/zmorge/transducers/zmorge-20150315-smor_newlemma.ca.zip'
unzip -u zmorge-20150315-smor_newlemma.ca.zip
#$WGET 'https://www.cis.uni-muenchen.de/~schmid/tools/RNNTagger/data/RNNTagger-1.3.zip'
#unzip -uo RNNTagger-1.3.zip
# NOTE: the original RNNTagger v1.3 seems to be unavailaible. As replacement, use v1.4.2
$WGET 'https://www.cis.uni-muenchen.de/~schmid/tools/RNNTagger/data/RNNTagger-1.4.2.zip'
unzip -uo RNNTagger-1.4.2.zip
find ./RNNTagger/lib/ -type f ! -name '*german*' -delete # remove unncessesary models
test -f "rwtagger_models.zip" || $WGET 'http://www.redewiedergabe.de/models/models.zip' -O rwtagger_models.zip
unzip -u rwtagger_models.zip -d rwtagger_models
$WGET 'https://github.com/uhh-lt/neural-coref/releases/download/konvens/droc_incremental_no_segment_distance.mar'
unzip -u droc_incremental_no_segment_distance.mar model_droc_incremental_no_segment_distance_May02_17-32-58_1800.bin
test -f "eventclassifier.zip" || $WGET 'https://github.com/uhh-lt/event-classification/releases/download/v0.2/demo_model.zip' -O 'eventclassifier.zip'
unzip -u eventclassifier.zip -d eventclassifier_model
