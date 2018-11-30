FROM gitlab-dl.nvidia.com:5005/dgx/cuda:10.0-devel-ubuntu16.04--18.12

ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

ENV TENSORFLOW_VERSION 1.12.0+
LABEL com.nvidia.tensorflow.version="${TENSORFLOW_VERSION}"
ENV NVIDIA_TENSORFLOW_VERSION 18.12

ARG PYVER=3.5

# libboost-*-dev and cmake needed for OpenSeq2Seq CTC Decoder/KenLM
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        python$PYVER \
        python$PYVER-dev \
        rsync \
        swig \
        unzip \
        zip \
        zlib1g-dev \
        libboost-locale-dev \
        libboost-program-options-dev \
        libboost-system-dev \
        libboost-thread-dev \
        libboost-test-dev \
        cmake && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONIOENCODING=utf-8 \
    LC_ALL=C.UTF-8

RUN rm -f /usr/bin/python && \
    rm -f /usr/bin/python`echo $PYVER | cut -c1-1` && \
    ln -s /usr/bin/python$PYVER /usr/bin/python && \
    ln -s /usr/bin/python$PYVER /usr/bin/python`echo $PYVER | cut -c1-1`

# TF 1.0 upstream needs this symlink
RUN mkdir -p /usr/lib/x86_64-linux-gnu/include/ && \
     ln -s /usr/include/cudnn.h /usr/lib/x86_64-linux-gnu/include/cudnn.h

# If installing multiple pips, install pip2 last so that pip == pip2 when done.
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# nltk version specified per OpenSeq2Seq requirements
RUN DALI_VERSION=0.5.0 \
 && DALI_BUILD=39581 \
 && pip install --no-cache-dir --upgrade \
                --extra-index-url https://developer.download.nvidia.com/compute/redist \
                --extra-index-url http://sqrl/dldata/pip-simple --trusted-host sqrl \
        nvidia-dali==${DALI_VERSION} \
        numpy==1.14.5 \
        pexpect \
        psutil \
        nltk==3.2.5 \
        future \
        mock \
        portpicker \
        h5py \
        keras_preprocessing==1.0.5 \
        keras_applications==1.0.6

# other OpenSeq2Seq dependencies
RUN pip install --no-cache-dir --upgrade \
        resampy \
        python_speech_features \
        pandas==0.23.0 \
        six \
        mpi4py \
        librosa==0.6.1 \
        matplotlib \
        joblib==0.11 \
        sentencepiece
# sacrebleu does not install cleanly with python2.
RUN test ${PYVER%.*} -eq 2 || pip install --no-cache-dir --upgrade sacrebleu

# Set up Bazel.
# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
#   https://github.com/bazelbuild/bazel/issues/418
ENV BAZELRC /root/.bazelrc
RUN echo "startup --batch" >> $BAZELRC && \
    echo "build --spawn_strategy=standalone --genrule_strategy=standalone" >> $BAZELRC

RUN BAZEL_VERSION=0.15.0 && \
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

ENV CUDA_TOOLKIT_PATH=/usr/local/cuda \
    CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu \
    NCCL_INSTALL_PATH=/usr/lib/x86_64-linux-gnu \
    NCCL_HDR_PATH=/usr/include

# Build and install TF
RUN ./nvbuild.sh --testlist --python$PYVER

ENV TF_ADJUST_HUE_FUSED=1 \
    TF_ADJUST_SATURATION_FUSED=1 \
    TF_ENABLE_WINOGRAD_NONFUSED=1 \
    TF_AUTOTUNE_THRESHOLD=2 \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/tensorflow

# TensorBoard
EXPOSE 6006

# Horovod with fp16 patch
ENV HOROVOD_GPU_ALLREDUCE=NCCL \
    HOROVOD_NCCL_INCLUDE=/usr/include \
    HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu \
    HOROVOD_NCCL_LINK=SHARED \
    HOROVOD_WITHOUT_PYTORCH=1
RUN cd /opt/tensorflow/third_party/horovod && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so ./libcuda.so.1 && \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD && \
    python setup.py install && \
    python setup.py clean && \
    rm ./libcuda.so.1

# OpenSeq2Seq CTC Decoder & KenLM
RUN cd /opt/tensorflow/nvidia-examples/OpenSeq2Seq && \
    ./scripts/install_kenlm.sh && \
    cd /opt/tensorflow && \
    ln -s nvidia-examples/OpenSeq2Seq/ctc_decoder_with_lm ./ && \
    ./nvbuild.sh --configonly --python$PYVER && \
    bazel build $(cat nvbuildopts) \
        //ctc_decoder_with_lm:libctc_decoder_with_kenlm.so \
        //ctc_decoder_with_lm:generate_trie && \
    cp bazel-bin/ctc_decoder_with_lm/libctc_decoder_with_kenlm.so \
        bazel-bin/ctc_decoder_with_lm/generate_trie \
        ctc_decoder_with_lm/ && \
    bazel clean --expunge && \
    rm .tf_configure.bazelrc .bazelrc /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    rm -rf ${HOME}/.cache/bazel /tmp/*


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

