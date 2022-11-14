FROM ubuntu AS get_miniconda
SHELL ["/bin/bash", "-c"]
RUN apt-get update \
  && apt-get install -y \
    wget \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
RUN wget --progress=dot:giga -O /miniconda.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh \
  && bash /miniconda.sh -b -p /opt/conda \
  && rm -f /miniconda.sh

FROM ubuntu AS invokeai
SHELL [ "/bin/bash", "-c" ]
RUN echo "" > ~/.bashrc
RUN apt-get update \
  && apt-get install -y \
    --no-install-recommends \
    gcc \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    pip \
    python3 \
    python3-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
RUN git clone -b main https://github.com/y5labs/dreamdispersal-docker.git "/dreamdispersal" \
  && cp \
    "/dreamdispersal/configs/models.yaml.example" \
    "/dreamdispersal/configs/models.yaml" \
  && ln -sf \
    "/dreamdispersal/environments-and-requirements/environment-lin-cuda.yml" \
    "/dreamdispersal/environment.yml" \
  && ln -sf \
    /data/models/v1-5-pruned-emaonly.ckpt \
    "/dreamdispersal/models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt" \
  && ln -sf \
    /data/outputs/ \
    "/dreamdispersal/outputs"
WORKDIR "/dreamdispersal"
COPY --from=get_miniconda "/opt/conda" "/opt/conda"
RUN source "/opt/conda/etc/profile.d/conda.sh" \
  && conda init bash \
  && source ~/.bashrc \
  && conda env create \
    --name "dreamdispersal" \
  && rm -Rf ~/.cache \
  && conda clean -afy \
  && echo "conda activate dreamdispersal" >> ~/.bashrc
# /src/gfpgan/experiments/pretrained_models/GFPGANv1.4.pth
RUN source ~/.bashrc \
  && python scripts/preload_models.py \
    --no-interactive
COPY ./entrypoint.sh /
ENTRYPOINT [ "/entrypoint.sh" ]
