FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
RUN apt-get update -y && apt-get -y install \
    curl \
    python3 \
    python3-pip \
    python3-venv \
    python3-distutils \
    swi-prolog \
    sfst \
    unzip \
    wget \
    git

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  POETRY_NO_INTERACTION=1 \
  TOKENIZER_PARALLELISM=false

# adjust if needed, e.g. via --build-arg=USER=anton
ARG USER=llprouser
ARG UID=""

RUN if [ -z "${UID}" ]; then adduser ${USER} --home /docker_home --disabled-password --gecos ""; \
        else adduser ${USER} --uid ${UID} --home /docker_home --disabled-password --gecos ""; fi

RUN mkdir /LLpro
RUN chown extehrmanntraut /LLpro

USER extehrmanntraut
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${PATH}:/docker_home/.local/bin"


WORKDIR /LLpro
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
RUN poetry config virtualenvs.in-project true
RUN poetry install --no-root

COPY --chown=${USER} resources/ resources/
COPY --chown=${USER} prepare.sh prepare.sh
RUN sh prepare.sh

COPY --chown=${USER} llpro llpro
COPY --chown=${USER} build.py build.py
RUN poetry install

COPY --chown=${USER} bin/ bin/
RUN poetry run python -c 'from bin.llpro_cli import create_pipe; create_pipe();'

ARG INSTALL_APEX=0
WORKDIR /tmp
RUN if [ "$INSTALL_APEX" = "1" ] ; then echo 'Building Apex' \
    && git clone https://github.com/NVIDIA/apex.git \
    && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .; fi

WORKDIR /LLpro
RUN rm -f resources/RNNTagger-1.3.zip resources/eventclassifier.zip  resources/stss-se-scene-segmenter/model.tar.gz

ENV TRANSFORMERS_OFFLINE=1
ENTRYPOINT ["poetry", "run", "python", "bin/llpro_cli.py"]
