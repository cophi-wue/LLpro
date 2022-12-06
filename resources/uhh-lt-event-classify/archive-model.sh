#!/usr/bin/env sh
ARCHIVER=torch-model-archiver

show_usage() {
    echo "Usage $0 MODEL_NAME WEIGHTS [VERSION]"
    echo ""
    echo "Build model-archive using the config MODEL_NAME with weights from the WEIGHTS file."
    echo ""
}


if [ ! -z $3 ]
then
    VERSION=$3
else
    VERSION="0.0.1"
fi

if [ $# -lt 2 ]
then
    show_usage
    exit 1
fi

# This is a dumb workaround for https://github.com/pytorch/serve/issues/566
# We create a new directory and symlink in all the individual directories we would like to include

# Relevant implementation at:
# https://github.com/pytorch/serve/blob/38eed4703664175160304b9e9880fa40d8481f11/model-archiver/model_archiver/model_packaging_utils.py#L148-L161
TEMP_DIR=$(mktemp -d)
ln -s "$(pwd)/$2" $TEMP_DIR/model
ln -s "$(pwd)/$2/.hydra" $TEMP_DIR/config
ln -s "$(pwd)/event_classify" $TEMP_DIR

$ARCHIVER \
    --model-name $1 \
    --handler torch_serve/model_handler.py \
    --extra-files $TEMP_DIR,torch_serve \
    --archive-format default \
    --requirements-file requirements.txt \
    -v $VERSION \
    --runtime python3 \
    -f

rm -rf $TEMP_DIR
