#!/usr/bin/env sh
ARCHIVER=torch-model-archiver

show_usage() {
    echo "Usage $0 MODEL_NAME WEIGHTS MODEL_VARIANT [VERSION]"
    echo ""
    echo "Build model-archive using the config MODEL_NAME with weights from the WEIGHTS file."
    echo "MODEL_VARIANT is either 'c2f' or 'incremental'"
    echo ""
    echo "Control the bert-windowsize using the WINDOWSIZE environment variable"
}


if [ ! -z $4 ]
then
    VERSION=$3
else
    VERSION="0.0.1"
fi

if [ $# -lt 3 ]
then
    show_usage
    exit 1
fi

if [ $3 = "c2f" ]
then
    MODEL_VARIANT=torch_serve/c2f.py
elif [ $3 = "incremental" ]
then
    MODEL_VARIANT=torch_serve/incremental.py
else
    echo "Invalid model variant $3"
    exit 2
fi

echo "MODELNAME='$1'" > torch_serve/model_config.py

if [ -z $WINDOWSIZE ]
then
    WINDOWSIZE="384"
fi

echo "WINDOWSIZE=$WINDOWSIZE" >> torch_serve/model_config.py

$ARCHIVER \
    --model-name $1 \
    --model-file $MODEL_VARIANT \
    --serialized-file $2 \
    --extra-files $(ls -1p *.py | grep -v / | xargs echo | sed 's/ /,/g'),experiments.conf,local.conf,torch_serve \
    --handler torch_serve/model_handler.py \
    --archive-format default \
    --requirements-file requirements.txt \
    -v $VERSION \
    --runtime python3 \
    -f


echo '
"""
Is overwritten during archival process to configure model.
"""
' > torch_serve/model_config.py
