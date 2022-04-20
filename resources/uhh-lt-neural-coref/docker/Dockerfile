ARG TORCHSERVE_TAG=0.3.0-cpu
FROM pytorch/torchserve:${TORCHSERVE_TAG}

COPY requirements.txt .
USER root
RUN apt-get update -y
RUN apt-get install -y cmake build-essential pkg-config libgoogle-perftools-dev git
RUN pip install wheel && pip install -r requirements.txt
USER model-server
# Download the base models so launch is faster :)
RUN python -c "from transformers import BertModel; BertModel.from_pretrained('german-nlp-group/electra-base-german-uncased'); BertModel.from_pretrained('deepset/gelectra-large')"
