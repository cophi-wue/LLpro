FROM nvidia/cuda:11.4.1-base-ubuntu20.04

WORKDIR /LL-Pipeline

RUN apt-get update -y
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
RUN apt-get -y install \
    python3 \
    python3-pip \
    swi-prolog \
    sfst \
    unzip \
    wget

RUN mkdir resources
COPY prepare.sh .
COPY requirements.txt .
RUN sh prepare.sh
COPY . .

ENTRYPOINT ["python3", "main.py"]
