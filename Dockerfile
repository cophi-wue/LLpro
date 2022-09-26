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

COPY . .
RUN sh prepare.sh
RUN rm -rf resources/invero-xl-span-cuda-2.0.0.tar
RUN python3 -c 'import llpro.pipeline; llpro.pipeline.preload_all_modules();'


ARG INSTALL_APEX=0
WORKDIR /tmp
RUN if [ "$INSTALL_APEX" = "1" ] ; then echo 'Building Apex' \
    && git clone https://github.com/NVIDIA/apex.git \
    && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .; fi


WORKDIR /LL-Pipeline
ENV TRANSFORMERS_OFFLINE=1
ENV TOKENIZER_PARALLELISM=false

ENTRYPOINT ["python3", "main.py"]
