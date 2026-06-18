# Docker image for interformer:
FROM nvidia/cuda:12.6.0-base-ubuntu22.04
ARG username

RUN rm -f /etc/apt/sources.list.d/*.list

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /main
RUN mkdir /main/home
WORKDIR /main

RUN adduser --disabled-password --gecos '' --shell /bin/bash $username \
    && chown -R $username:$username /main
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER $username

ENV HOME=/main/home
RUN mkdir $HOME/.cache $HOME/.config \
    && chmod -R 777 $HOME

ENV PATH=$HOME/mambaforge/bin:$PATH
COPY environment.yml /main/environment.yml
RUN curl -sLo ~/mambaforge.sh https://github.com/conda-forge/miniforge/releases/download/4.12.0-2/Mambaforge-4.12.0-2-Linux-x86_64.sh \
    && chmod +x ~/mambaforge.sh \
    && ~/mambaforge.sh -b -p ~/mambaforge \
    && rm ~/mambaforge.sh \
    && mamba env update -n base -f /main/environment.yml \
    && mamba clean -ya

# Version 3&4 - with plip and extensions for wandb:
FROM blakshana/interformer:v2

RUN /main/home/mambaforge/bin/mamba install -c conda-forge \
python-dotenv \
wandb \
-y

# Version 5&6 - for pyvina_core installation with c++ compiler:
FROM blakshana/interformer:v4

USER root
RUN apt-get update && apt-get install -y \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
USER interformer

# Version 6:
FROM blakshana/interformer:v4

USER root
RUN apt-get update && apt-get install -y \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
USER interformer

RUN /main/home/mambaforge/bin/mamba install -c conda-forge \
    boost \
    -y
    