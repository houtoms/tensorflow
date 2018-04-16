FROM gitlab-dl.nvidia.com:5005/dgx/cuda:9.0-cudnn7.1-devel-ubuntu16.04--18.05

################################################################################
# TODO: REMOVE THESE LINES ONCE BASE CONTIANER INTEGRATES MOFED USERSPACE DRIVER
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        libnl-route-3-200 \
        libnuma1 && \
    rm -rf /var/lib/apt/lists/*

ENV MOFED_VERSION=3.4-1.0.0.0
RUN wget -q -O - http://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VERSION}/MLNX_OFED_LINUX-${MOFED_VERSION}-ubuntu16.04-x86_64.tgz | tar -xzf - && \
        dpkg --install MLNX_OFED_LINUX-${MOFED_VERSION}-ubuntu16.04-x86_64/DEBS/libibverbs1_*_amd64.deb && \
        dpkg --install MLNX_OFED_LINUX-${MOFED_VERSION}-ubuntu16.04-x86_64/DEBS/libibverbs-dev_*_amd64.deb && \
        dpkg --install MLNX_OFED_LINUX-${MOFED_VERSION}-ubuntu16.04-x86_64/DEBS/libmlx5-1_*_amd64.deb && \
        dpkg --install MLNX_OFED_LINUX-${MOFED_VERSION}-ubuntu16.04-x86_64/DEBS/ibverbs-utils_*_amd64.deb && \
        rm -rf MLNX_OFED_LINUX-${MOFED_VERSION}-ubuntu16.04-x86_64
################################################################################

ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

ENV TENSORFLOW_VERSION 1.7.0+
LABEL com.nvidia.tensorflow.version="${TENSORFLOW_VERSION}"
ENV NVIDIA_TENSORFLOW_VERSION 18.05

ARG PYVER=2.7

RUN apt-get update && apt-get install -y --no-install-recommends \
        libhwloc-dev \
        libnuma-dev \
        pkg-config \
        python$PYVER \
        python$PYVER-dev \
        rsync \
        swig \
        unzip \
        zip \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONIOENCODING utf-8
ENV LC_ALL C.UTF-8
RUN rm -f /usr/bin/python && \
    rm -f /usr/bin/python`echo $PYVER | cut -c1-1` && \
    ln -s /usr/bin/python$PYVER /usr/bin/python && \
    ln -s /usr/bin/python$PYVER /usr/bin/python`echo $PYVER | cut -c1-1`

# Needed for Horovod
RUN OPENMPI_VERSION=3.0.0 && \
    wget -q -O - https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-${OPENMPI_VERSION}.tar.gz | tar -xzf - && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --enable-orterun-prefix-by-default --with-cuda --with-verbs \
                --prefix=/usr/local/mpi --disable-getpwuid && \
    make -j"$(nproc)" install && \
    cd .. && rm -rf openmpi-${OPENMPI_VERSION} && \
    echo "/usr/local/mpi/lib" >> /etc/ld.so.conf.d/openmpi.conf && ldconfig
ENV PATH /usr/local/mpi/bin:$PATH

# The following works around a segfault in OpenMPI 3.0
# when run within a single node without ssh being installed.
RUN /bin/echo -e '#!/bin/bash'\
'\ncat <<EOF'\
'\n======================================================================'\
'\nTo run a multi-node job, install an ssh client and clear plm_rsh_agent'\
'\nin '/usr/local/mpi/etc/openmpi-mca-params.conf'.'\
'\n======================================================================'\
'\nEOF'\
'\nexit 1' >> /usr/local/mpi/bin/rsh_warn.sh && \
    chmod +x /usr/local/mpi/bin/rsh_warn.sh && \
    echo "plm_rsh_agent = /usr/local/mpi/bin/rsh_warn.sh" >> /usr/local/mpi/etc/openmpi-mca-params.conf

# TF 1.0 upstream needs this symlink
RUN mkdir -p /usr/lib/x86_64-linux-gnu/include/ && \
     ln -s /usr/include/cudnn.h /usr/lib/x86_64-linux-gnu/include/cudnn.h

# If installing multiple pips, install pip2 last so that pip == pip2 when done.
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip install --no-cache-dir --upgrade numpy==1.11.0 pexpect psutil nltk future

# Set up Bazel.
# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
#   https://github.com/bazelbuild/bazel/issues/418
ENV BAZELRC /root/.bazelrc
RUN echo "startup --batch" >> $BAZELRC && \
    echo "build --spawn_strategy=standalone --genrule_strategy=standalone" >> $BAZELRC

RUN BAZEL_VERSION=0.11.0 && \
    mkdir /bazel && cd /bazel && \
    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    bash ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    rm -rf /bazel

# TensorRT
RUN wget -qO libnvinfer.deb "http://cuda-repo/release-candidates/Libraries/TensorRT/v3.0/3.0.4-GA-cl_23531598/CUDA9.0-r384/Ubuntu16_04-x64/deb/libnvinfer4_4.0.4-1+cuda9.0_amd64.deb" && \
    wget -qO libnvinfer-dev.deb "http://cuda-repo/release-candidates/Libraries/TensorRT/v3.0/3.0.4-GA-cl_23531598/CUDA9.0-r384/Ubuntu16_04-x64/deb/libnvinfer-dev_4.0.4-1+cuda9.0_amd64.deb" && \
    dpkg -i libnvinfer.deb && \
    dpkg -i libnvinfer-dev.deb && \
    rm /usr/lib/x86_64-linux-gnu/libnvcaffe_parser.a \
       /usr/lib/x86_64-linux-gnu/libnvparsers.a \
       /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.a \
       /usr/lib/x86_64-linux-gnu/libnvinfer.a \
       libnvinfer.deb libnvinfer-dev.deb

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
ENV TF_CUDA_COMPUTE_CAPABILITIES "5.2,6.0,6.1,7.0"
ENV TF_NEED_GCP 0
ENV TF_NEED_HDFS 0
ENV TF_ENABLE_XLA 1
ENV TF_NEED_TENSORRT 1
ENV CC_OPT_FLAGS "-march=sandybridge -mtune=broadwell"

# Build and install TF
RUN ./nvbuild.sh --python$PYVER

ENV TF_ADJUST_HUE_FUSED         1
ENV TF_ADJUST_SATURATION_FUSED  1
ENV TF_ENABLE_WINOGRAD_NONFUSED 1
ENV TF_AUTOTUNE_THRESHOLD       2

# TensorBoard
EXPOSE 6006

# Horovod with fp16 patch
ENV HOROVOD_GPU_ALLREDUCE NCCL
ENV HOROVOD_NCCL_INCLUDE /usr/include
ENV HOROVOD_NCCL_LIB /usr/lib/x86_64-linux-gnu
ENV HOROVOD_NCCL_LINK SHARED
ENV HOROVOD_GPU_ALLGATHER MPI
ENV HOROVOD_GPU_BROADCAST MPI
RUN git clone https://github.com/uber/horovod.git /horovod && \
    cd /horovod && \
    git checkout 85882b0b04afe207ae9d5b91cc1eef726dea0083 && \
    git apply /opt/tensorflow/horovod_fp16.patch && \
    git apply /opt/tensorflow/horovod_shared_lib.patch && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so ./libcuda.so.1 && \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD && \
    python setup.py install && \
    rm -rf /horovod

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
