FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg \
    sox libsox-dev libsox-fmt-all curl build-essential

# install rust and cargo (required for tokenizers)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN python3 -m pip install --no-cache-dir --upgrade pip

# 4. Copy and Install Requirements (With Force-Reinstall)
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade --force-reinstall -r requirements.txt
