FROM nvdl.githost.io:4678/dgx/tensorflow:17.02-stage
MAINTAINER NVIDIA CORPORATION <cudatools@nvidia.com>

WORKDIR /opt

COPY src/cnn_benchmarks/ ./cnn_benchmarks

CMD cd cnn_benchmarks && \
    bash run2.sh
