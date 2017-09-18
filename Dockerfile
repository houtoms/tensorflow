FROM nvdl.githost.io:4678/dgx/cuda:9.0-cudnn7-devel-ubuntu16.04--17.10

ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

ENV TENSORFLOW_VERSION 1.3.0
LABEL com.nvidia.tensorflow.version="${TENSORFLOW_VERSION}"
ENV NVIDIA_TENSORFLOW_VERSION 17.10

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        swig \
        unzip \
        zip \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# TF 1.0 upstream needs this symlink
RUN mkdir -p /usr/lib/x86_64-linux-gnu/include/ && \
     ln -s /usr/include/cudnn.h /usr/lib/x86_64-linux-gnu/include/cudnn.h

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip install --upgrade --no-cache-dir numpy==1.11.0 pexpect psutil

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

RUN BAZEL_VERSION=0.5.0 && \
    mkdir /bazel && cd /bazel && \
    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    bash ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    rm -rf /bazel

# Download and build TensorFlow.
WORKDIR /opt/tensorflow
COPY . .

# Link examples to workspace
RUN mkdir -p /workspace/nvidia-examples && \
     ln -s /opt/tensorflow/nvidia-examples/* /workspace/nvidia-examples/

# HACK to enable use of system nccl.h when building TF
RUN mkdir -p nccl/src && \
    cp /usr/include/nccl.h nccl/src/ && \
    cp `ls /usr/share/doc/libnccl?/copyright | tail -1` nccl/LICENSE.txt && \
    tar -cf third_party/nccl.tar nccl/ && \
    gzip    third_party/nccl.tar && \
    rm -r nccl/

ENV CUDA_TOOLKIT_PATH /usr/local/cuda
ENV TF_CUDA_VERSION "9.0"
ENV TF_CUDNN_VERSION "7"
ENV CUDNN_INSTALL_PATH /usr/lib/x86_64-linux-gnu
ENV TF_NEED_CUDA 1
ENV TF_CUDA_COMPUTE_CAPABILITIES "3.5,5.2,6.0,6.1,7.0"
ENV TF_NEED_GCP 0
ENV TF_NEED_HDFS 0
ENV TF_ENABLE_XLA 1
ENV PYTHON_BIN_PATH /usr/bin/python
RUN yes "" | ./configure
RUN bazel fetch "//tensorflow/... -//tensorflow/contrib/nccl/... -//tensorflow/examples/android/..." && \
    bash third_party/patch_eigen_for_cuda9.sh
RUN bazel build -c opt --config=cuda tensorflow/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
    pip install --upgrade /tmp/pip/tensorflow-*.whl && \
    rm -rf /tmp/pip/tensorflow-*.whl && \
    bazel clean --expunge

ENV TF_ADJUST_HUE_FUSED         1
ENV TF_ADJUST_SATURATION_FUSED  1
ENV TF_ENABLE_WINOGRAD_NONFUSED 1
ENV TF_AUTOTUNE_THRESHOLD       2
ENV CUDA_DEVICE_MAX_CONNECTIONS 12 # WAR for perf bug on K80 + NCCL

# TensorBoard
EXPOSE 6006

WORKDIR /workspace
COPY NVREADME.md README.md
COPY docker-examples docker-examples
RUN chmod -R a+w /workspace

COPY nvidia_entrypoint.sh /usr/local/bin
ENTRYPOINT ["/usr/local/bin/nvidia_entrypoint.sh"]

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID ${NVIDIA_BUILD_ID:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
ARG NVIDIA_BUILD_REF
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"
