# hash:sha256:f0416e65919e52c5e078c326849576da2be6eaf33f20c32433c4c6de590ee534
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
        software-properties-common=0.99.22.9 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    aind-data-schema==2.5.0 \
    aind-data-schema-models==5.4.2 \
    aind-metadata-manager==0.0.6 \
    hdmf-zarr==0.11.3 \
    jupyterlab==4.1.6 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    pynwb==3.1.2
