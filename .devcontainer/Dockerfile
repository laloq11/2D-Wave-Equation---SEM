FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
# FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    ca-certificates \
    libcli11-dev \
    libgomp1 \
    python3 \
    python3-dev \
    python3-pip \
    openssh-client \
    gdb \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy pandas matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -fsSL https://claude.ai/install.sh | bash \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace
