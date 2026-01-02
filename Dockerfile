# -------------------------- Build-time arguments (defaults) --------------------------
ARG DEEPSTREAM_TAG=8.0-gc-triton-devel

# -------------------------- Base image --------------------------
FROM nvcr.io/nvidia/deepstream:${DEEPSTREAM_TAG}

ARG DEEPSTREAM_TAG
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /dinov3_deepstream

# Copy repo
COPY . /dinov3_deepstream

# -------------------------- Extra tooling --------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        build-essential \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*



