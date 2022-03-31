FROM python:3.7.13-bullseye

WORKDIR /LL-Pipeline

RUN apt-get update -y
RUN apt-get -y install \
    python3-pip \
    swi-prolog \
    sfst

COPY requirements.txt .
RUN pip3 install --no-cache-dir -vvv -r requirements.txt
RUN python3 -c 'import nltk; nltk.download("punkt")'
COPY . .

ENTRYPOINT ["python3", "main.py"]
