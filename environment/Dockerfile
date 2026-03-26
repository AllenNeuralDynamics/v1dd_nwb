# hash:sha256:c6b648556d03686e4c0a3d4d6773b82abbcdb76c00b997d7ef21de3b38d5d84a
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip3 install -U --no-cache-dir \
    aind-data-schema \
    aind-metadata-manager \
    hdmf-zarr \
    pandas \
    pynwb
