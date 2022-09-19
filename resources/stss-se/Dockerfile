FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

## config
ARG USER=zehe
ARG UID=1052
# set to 1 to install conda:
ARG INSTALL_CONDA=1

## setup
RUN touch `date` && apt-get update

RUN apt install python3 python3-pip python3-dev python3-venv zsh byobu htop vim git wget lsof fuse parallel rsync -y zip unzip

RUN mkdir /docker_home

RUN adduser ${USER} --uid ${UID} --home /docker_home/ls6/${USER}/ --disabled-password --gecos "" --no-create-home
RUN mkdir -p /docker_home/ls6/${USER}
RUN chown -R ${USER} /docker_home/ls6/${USER}

RUN mkdir -p /pip
RUN chown -R ${USER} /pip


USER ${UID}
RUN python3 -m venv /pip


ADD requirements.txt .
RUN python3 -m venv /pip
## not sure yet, but I could not install the required libraries without expliciyl updating the following libraries
RUN bash -c "source /pip/bin/activate && pip3 install -U pip setuptools wheel"
RUN bash -c "source /pip/bin/activate && pip3 install --upgrade cython"
RUN bash -c "source /pip/bin/activate && pip3 install numpy"
## to download the pretrained model
RUN bash -c "source /pip/bin/activate && pip3 install gdown"

RUN bash -c "source /pip/bin/activate && pip3 install -r requirements.txt"

ENV PYTHONUNBUFFERED=1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

USER 0

RUN if [ "${INSTALL_CONDA}" = "1" ]; then bash -c 'mkdir /conda && chown ${UID} conda && \
\
cd /conda && \
/bin/bash -c "wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O ./miniconda.sh" && \
chmod 0755 ./miniconda.sh && \
/bin/bash -c "./miniconda.sh -b -p ./conda" && \
\
ln -s "/conda/conda/bin/conda" "/usr/local/bin/conda" && \
rm ./miniconda.sh && \
\
chown -R ${UID} /conda'; \
fi

RUN mkdir -p /data/train
RUN mkdir -p /data/test

RUN mkdir -p /predictions
RUN mkdir -p /code

RUN chown -R ${USER} /data
RUN chown -R ${USER} /predictions
RUN chown -R ${USER} /code
## download the pretrained model
RUN bash -c "source /pip/bin/activate && gdown --id 1yayKtOT2pGD7YQpL-r9p3D-ErMt6rVeR"
## download the pretrained model
USER ${UID}

ADD eval.py /
ADD predict.py /
ADD code/ code/

ENTRYPOINT bash -c "source /pip/bin/activate && python /predict.py"