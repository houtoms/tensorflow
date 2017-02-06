FROM nvdl.githost.io:4678/dgx/tensorflow:17.02-stage
MAINTAINER NVIDIA CORPORATION <cudatools@nvidia.com>

RUN pip install future

RUN git clone --depth 1 https://github.com/soumith/convnet-benchmarks.git /opt/convnet-benchmarks

WORKDIR /opt/convnet-benchmarks/tensorflow

CMD python benchmark_alexnet.py
