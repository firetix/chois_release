# syntax=docker/dockerfile:1.7
#
# CHOIS (ECCV 2024) docker image.
# Matches the paper's environment (Ubuntu 20.04, Python 3.8, CUDA 11.3, PyTorch 1.11)
# and includes Blender for the provided demo scripts.
#
# Note: The base image is linux/amd64. If you're on Apple Silicon, Docker will use emulation.
FROM --platform=linux/amd64 nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.8 python3.8-dev python3-pip \
      git wget ca-certificates xz-utils \
      ffmpeg \
      build-essential cmake ninja-build pkg-config \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      libx11-6 libxi6 libxfixes3 libxrandr2 libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*

# Make `python` available (the upstream scripts use `python ...`).
RUN ln -sf /usr/bin/python3.8 /usr/local/bin/python \
    && python -m pip install --upgrade pip setuptools wheel

# PyTorch + CUDA 11.3 (matches README).
RUN python -m pip install --no-cache-dir \
      torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
      --extra-index-url https://download.pytorch.org/whl/cu113

# Blender (headless rendering via `-b`).
ARG BLENDER_VERSION=3.6.3
RUN wget -q "https://download.blender.org/release/Blender3.6/blender-${BLENDER_VERSION}-linux-x64.tar.xz" -O /tmp/blender.tar.xz \
    && tar -xf /tmp/blender.tar.xz -C /opt \
    && rm /tmp/blender.tar.xz \
    && ln -sf "/opt/blender-${BLENDER_VERSION}-linux-x64/blender" /usr/local/bin/blender

WORKDIR /workspace

# Install python dependencies first for better layer caching.
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir \
        tqdm dotmap PyYAML omegaconf loguru \
        fvcore iopath \
        transforms3d smplx \
    && python -m pip install --no-cache-dir pytorch3d \
        -f "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html" \
    && python -m pip install --no-cache-dir "git+https://github.com/openai/CLIP.git" \
    && python -m pip install --no-cache-dir "git+https://github.com/otaheri/chamfer_distance.git" \
    && python -m pip install --no-cache-dir "git+https://github.com/otaheri/bps_torch.git" \
    # human_body_prior upstream has moved to newer Python/Torch. Pin a commit compatible with this repo's API
    # (BodyModel supports `bm_fname=...` and `num_expressions=None`) and ignore its old deps.
    && python -m pip install --no-cache-dir --no-deps "git+https://github.com/nghorbani/human_body_prior.git@362a904"

# Copy repo last (so edits don't invalidate the dependency layer cache).
COPY . /workspace

# Reasonable defaults; can be overridden via `docker-compose.yml` env.
ENV PYTHONPATH=/workspace \
    BLENDER_PATH=blender \
    BLENDER_UTILS_ROOT_FOLDER=/workspace/manip/vis \
    BLENDER_SCENE_FOLDER=/workspace/processed_data/blender_files \
    CHOIS_BLENDER_SCENE_FOLDER=/workspace/processed_data/blender_files \
    SMPL_ALL_MODELS_DIR=/workspace/processed_data/smpl_all_models \
    SMPLH_PATH=/workspace/processed_data/smpl_all_models/smplh_amass \
    TORCH_HOME=/workspace/.cache/torch \
    WANDB_MODE=disabled

CMD ["bash"]
