#!/usr/bin/env sh
mkdir -p data
curl https://zenodo.org/record/6414926/files/forTEXT/EvENT_Dataset-v.1.1.zip?download=1 -o data/event_data.zip
unzip data/event_data.zip -d data
