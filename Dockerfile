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


COPY . .
RUN sh prepare.sh
RUN rm -rf resources/invero-xl-span-cuda-2.0.0.tar

ARG INSTALL_APEX=0
WORKDIR /tmp
RUN if [ "$INSTALL_APEX" = "1" ] ; then echo 'Building Apex' \
    && git clone https://github.com/NVIDIA/apex.git \
    && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .; fi


WORKDIR /LL-Pipeline

ENTRYPOINT ["python3", "main.py"]
