FROM nvdl.githost.io:4678/dgx/cuda:8.0-cudnn6-devel-ubuntu14.04--17.02

ENV TENSORFLOW_VERSION 0.12.0-dev
LABEL com.nvidia.tensorflow.version="${TENSORFLOW_VERSION}"
ENV NVIDIA_TENSORFLOW_VERSION 17.02

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID ${NVIDIA_BUILD_ID:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
ARG NVIDIA_BUILD_REF
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        swig \
        unzip \
        zip \
        zlib1g-dev \
        vim && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip install --upgrade --no-cache-dir numpy

# Set up Bazel.
RUN add-apt-repository -y ppa:openjdk-r/ppa && apt-get update && \
    apt-get install -y --no-install-recommends openjdk-8-jdk openjdk-8-jre-headless && \
    rm -rf /var/lib/apt/lists/*

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
#   https://github.com/bazelbuild/bazel/issues/418
ENV BAZELRC /root/.bazelrc
RUN echo "startup --batch" >> $BAZELRC && \
    echo "build --spawn_strategy=standalone --genrule_strategy=standalone" >> $BAZELRC

RUN BAZEL_VERSION=0.4.2 && \
    mkdir /bazel && cd /bazel && \
    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE.txt && \
    bash ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    rm -rf /bazel

# Download and build TensorFlow.
WORKDIR /opt/tensorflow
COPY . .

ENV CUDA_TOOLKIT_PATH /usr/local/cuda
ENV CUDNN_INSTALL_PATH /usr/lib/x86_64-linux-gnu
ENV TF_NEED_CUDA 1
ENV TF_CUDA_COMPUTE_CAPABILITIES "3.5,5.2,6.0,6.1"
ENV TF_NEED_GCP 0
ENV TF_NEED_HDFS 0
RUN yes "" | ./configure && \
    bazel build -c opt --config=cuda tensorflow/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
    pip install --upgrade /tmp/pip/tensorflow-*.whl && \
    bazel clean --expunge

# TensorBoard
EXPOSE 6006

WORKDIR /workspace
COPY NVREADME.md README.md
RUN chmod -R a+w /workspace

COPY nvidia_entrypoint.sh /usr/local/bin
ENTRYPOINT ["/usr/local/bin/nvidia_entrypoint.sh"]
