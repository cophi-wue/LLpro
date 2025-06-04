#!/usr/bin/bash
set -e
WGET="wget --no-clobber --show-progress --progress=bar:force:noscroll"

cd resources
mkdir -p tmp
$WGET 'https://corpora.linguistik.uni-erlangen.de/someweta/german_newspaper_2020-05-28.model'
$WGET 'https://pub.cl.uzh.ch/users/sennrich/zmorge/transducers/zmorge-20150315-smor_newlemma.ca.zip'
unzip -u zmorge-20150315-smor_newlemma.ca.zip
#$WGET 'https://www.cis.uni-muenchen.de/~schmid/tools/RNNTagger/data/RNNTagger-1.3.zip'
#unzip -uo RNNTagger-1.3.zip
# NOTE: the original RNNTagger v1.3 seems to be unavailaible. As replacement, use v1.4.7
# (I really should think about a different solution...)
$WGET 'https://www.cis.uni-muenchen.de/~schmid/tools/RNNTagger/data/RNNTagger-1.4.7.zip'
unzip -uo RNNTagger-1.4.7.zip
find ./RNNTagger/lib/ -type f ! -name '*german*' -delete # remove unncessesary models
$WGET 'https://github.com/uhh-lt/neural-coref/releases/download/konvens/droc_incremental_no_segment_distance.mar'
unzip -u droc_incremental_no_segment_distance.mar model_droc_incremental_no_segment_distance_May02_17-32-58_1800.bin
$WGET 'https://github.com/uhh-lt/neural-coref/releases/download/konvens/droc_c2f.mar'
unzip -u droc_c2f.mar model_droc_c2f_May12_17-38-53_1800.bin
test -f "eventclassifier.zip" || $WGET 'https://github.com/uhh-lt/event-classification/releases/download/v0.2/demo_model.zip' -O 'eventclassifier.zip'
unzip -u eventclassifier.zip -d eventclassifier_model

if [[ "$LLPRO_EXPERIMENTAL" -eq 1 ]]; then
    test -f "emotions_models.zip" || $WGET 'https://owncloud.gwdg.de/index.php/s/g2PjWWcknSRlMSd/download' -O emotions_models.zip
    mkdir -p konle_emotion_weights
    unzip -u emotions_models.zip -d konle_emotion_weights
    mv --verbose konle_emotion_weights/models/* konle_emotion_weights
    rmdir konle_emotion_weights/models
fi
