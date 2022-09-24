#!/usr/bin/bash
set -e
WGET="wget --no-clobber --show-progress --progress=bar:force:noscroll"

pip3 install --no-cache-dir -r requirements.txt

cd resources
$WGET 'https://corpora.linguistik.uni-erlangen.de/someweta/german_newspaper_2020-05-28.model'
$WGET 'https://pub.cl.uzh.ch/users/sennrich/zmorge/transducers/zmorge-20150315-smor_newlemma.ca.zip'
unzip zmorge-20150315-smor_newlemma.ca.zip
$WGET 'https://www.cis.uni-muenchen.de/~schmid/tools/RNNTagger/data/RNNTagger-1.3.zip'
unzip -uo RNNTagger-1.3.zip
find ./RNNTagger/lib/ -type f ! -name '*german*' -delete # remove unncessesary models
test -f "rwtagger_models.zip" || $WGET 'http://www.redewiedergabe.de/models/models.zip' -O rwtagger_models.zip
unzip rwtagger_models.zip -d rwtagger_models
$WGET 'https://github.com/uhh-lt/neural-coref/releases/download/konvens/droc_incremental_no_segment_distance.mar'
unzip droc_incremental_no_segment_distance.mar model_droc_incremental_no_segment_distance_May02_17-32-58_1800.bin
test -f "./stss-se/model.tar.gz" || gdown 1yayKtOT2pGD7YQpL-r9p3D-ErMt6rVeR -O ./stss-se/model.tar.gz
tar xf ./stss-se/model.tar.gz -C ./stss-se/extracted_model

invero_tarfile="invero-xl-span-cuda-2.0.0.tar"
if [ -f "$invero_tarfile" ]; then
    if tar tf "$invero_tarfile" manifest.json >/dev/null; then
        echo >&2 'Found invero image'
    else
        echo >&2 "WARNING: resources folder contains $invero_tarfile but is not a Docker image!"
        echo >&2 "You might need to untar the archive to extract the Docker image"
        exit 1
    fi
    tar xf "$invero_tarfile" \
        dfd58a13d4e570b36f9cf854db2b874261d06617729eff20d6634db0d15c24d0/layer.tar -O \
    | tar xf - -C ./inveroxl/resources/model --strip-components 3 \
        app/resources/model
else
    echo >&2 "WARNING: Docker image $invero_tarfile not found - pipeline will be unable to run semantic role labeling"
fi

cd ..
python3 -c 'import llpro.pipeline; llpro.pipeline.preload_all_modules();'
