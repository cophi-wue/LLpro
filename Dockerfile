FROM nvidia/cuda:11.4.1-base-ubuntu20.04

WORKDIR /LL-Pipeline

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
RUN apt-get update -y && apt-get -y install \
    python3 \
    python3-pip \
    swi-prolog \
    sfst \
    unzip \
    wget


COPY . .
RUN sh prepare.sh
RUN rm -rf resources/invero-xl-span-cuda-2.0.0.tar

ENTRYPOINT ["python3", "main.py"]
