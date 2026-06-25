FROM docker.io/gnina/gnina:latest

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/envs/gninatorch_env/bin:$CONDA_DIR/bin:$PATH
ENV CONDA_DEFAULT_ENV=gninatorch_env

# basic tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip \
        wget git ca-certificates build-essential \
    && pip3 install --no-cache-dir pandas \
    && rm -rf /var/lib/apt/lists/*

# miniforge/conda installation
RUN wget -qO /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p $CONDA_DIR \
    && rm /tmp/miniforge.sh \
    && conda clean -afy

# clone gnina-torch
WORKDIR /opt
RUN git clone https://github.com/RMeli/gnina-torch.git

# create gninatorch environment
WORKDIR /opt/gnina-torch
RUN conda create -y -n gninatorch_env -c pytorch -c nvidia -c conda-forge \
    python=3.12 \
    pytorch \
    pytorch-cuda=12.1 \
    pytorch-ignite \
    "numpy<2" \
    scipy \
    pandas \
    scikit-learn \
    matplotlib \
    openbabel \
    mlflow \
    pip \
    && conda clean -afy


# install matplotlib and molgrid

RUN conda run -n gninatorch_env python -m pip install "numpy<2"

RUN conda run -n gninatorch_env python -m pip install molgrid

# patch gnina-torch training.py to export a full model file
RUN python - <<'PY'
from pathlib import Path

p = Path("/opt/gnina-torch/gninatorch/training.py")
text = p.read_text()

old = "trainer.run(train_loader, max_epochs=args.iterations)"
new = old + """

    # Export standalone TorchScript model for GNINA --cnn_model
    model.eval()
    model.cpu()
    model_outfile = os.path.join(args.out_dir, "gnina_retrained_full_model.pt")
    torch.jit.save(model, model_outfile)
    print(f"Saved GNINA-readable TorchScript model to: {model_outfile}")
"""

if old not in text:
    raise RuntimeError("Could not find trainer.run line for model export patch")

text = text.replace(old, new)
p.write_text(text)
PY

# install gninatorch

RUN conda run -n gninatorch_env python -m pip install .

# workspace dirs
WORKDIR /workspace

COPY run_proto_gnina.py /workspace/run_proto_gnina.py
COPY run_gnina.sh /workspace/run_gnina.sh

# run shell
RUN chmod +x /workspace/run_gnina.sh

# debug help
RUN gnina --help >/dev/null \
    && conda run -n gninatorch_env obrms --help >/dev/null \
    && conda run -n gninatorch_env python -c "import numpy; print(numpy.__version__); import torch; import molgrid; import gninatorch; import mlflow; print('gninatorch + mlflow ok')"

RUN conda run -n gninatorch_env python -c "import gninatorch"
RUN conda run -n gninatorch_env python -m gninatorch.training --help

CMD ["/workspace/run_gnina.sh"]