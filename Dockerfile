FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

WORKDIR /LL-Pipeline

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
RUN apt-get update -y && apt-get -y install \
    python3 \
    python3-pip \
    swi-prolog \
    sfst \
    unzip \
    wget \
    git


COPY requirements.txt requirements.txt
RUN sh -c 'pip3 install --no-cache-dir -r requirements.txt'

COPY resources resources
COPY prepare.sh prepare.sh
RUN sh prepare.sh

COPY main.py main.py
COPY llpro llpro
RUN python3 -c 'from main import create_pipe; create_pipe();'



ARG INSTALL_APEX=0
WORKDIR /tmp
RUN if [ "$INSTALL_APEX" = "1" ] ; then echo 'Building Apex' \
    && git clone https://github.com/NVIDIA/apex.git \
    && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .; fi


WORKDIR /LL-Pipeline
ENV TRANSFORMERS_OFFLINE=1
ENV TOKENIZER_PARALLELISM=false

ENTRYPOINT ["python3", "-u", "main.py"]
