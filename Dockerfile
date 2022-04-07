FROM python:3.7.13-bullseye

WORKDIR /LL-Pipeline

RUN apt-get update -y
RUN apt-get -y install \
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
