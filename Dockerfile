ARG FROM_IMAGE_NAME=gitlab-master.nvidia.com:5005/dl/dgx/cuda:10.1-devel-ubuntu16.04--master
FROM ${FROM_IMAGE_NAME}

ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

ARG TENSORFLOW_VERSION
ENV TENSORFLOW_VERSION=${TENSORFLOW_VERSION}
LABEL com.nvidia.tensorflow.version="${TENSORFLOW_VERSION}"
ARG NVIDIA_TENSORFLOW_VERSION
ENV NVIDIA_TENSORFLOW_VERSION=${NVIDIA_TENSORFLOW_VERSION}

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
RUN pip install --no-cache-dir --upgrade \
        numpy==1.14.5 \
        pexpect==4.6.0 \
        psutil==5.6.1 \
        nltk==3.2.5 \
        future==0.17.1 \
        jupyterlab==$(test ${PYVER%.*} -eq 2 && echo 0.33.12 || echo 0.35.4) \
        mock==2.0.0 \
        portpicker==1.3.1 \
        h5py==2.9.0 \
        keras_preprocessing==1.0.5 \
        keras_applications==1.0.6

# other OpenSeq2Seq dependencies
RUN pip install --no-cache-dir --upgrade \
        resampy==0.2.1 \
        numba==0.43.0 \
        llvmlite==0.28.0 \
        python_speech_features==0.6 \
        pandas==0.23.0 \
        six==1.12.0 \
        mpi4py==3.0.1 \
        librosa==0.6.1 \
        matplotlib==$(test ${PYVER%.*} -eq 2 && echo 2.2.4 || echo 3.0.3) \
        joblib==0.11 \
        sentencepiece==0.1.6 \
        $(test ${PYVER%.*} -eq 3 && echo "sacrebleu==1.2.20" || echo "")

# Set up Bazel.
# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
#   https://github.com/bazelbuild/bazel/issues/418
ENV BAZELRC /root/.bazelrc
RUN echo "startup --batch" >> $BAZELRC && \
    echo "build --spawn_strategy=standalone --genrule_strategy=standalone" >> $BAZELRC

RUN BAZEL_VERSION=0.19.2 && \
    mkdir /bazel && cd /bazel && \
    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    bash ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    rm -rf /bazel

# Download and build TensorFlow.
WORKDIR /opt/tensorflow
COPY . .

# MKL-DNN patch for floating point exception bug
# Create mkl-dnn-${MKL_DNN_COMMIT}-patched.tar.gz for use by tensorflow/workspace.bzl
RUN MKL_DNN_COMMIT=733fc908874c71a5285043931a1cf80aa923165c && \
    curl -fSsL -O https://mirror.bazel.build/github.com/intel/mkl-dnn/archive/${MKL_DNN_COMMIT}.tar.gz && \
    tar -xf ${MKL_DNN_COMMIT}.tar.gz && \
    rm -f ${MKL_DNN_COMMIT}.tar.gz && \
    cd mkl-dnn-${MKL_DNN_COMMIT} && \
    patch -p0 < ../mkl-dnn.patch && \
    cd .. && \
    tar --mtime='1970-01-01' -cf mkl-dnn-${MKL_DNN_COMMIT}-patched.tar mkl-dnn-${MKL_DNN_COMMIT}/ && \
    rm -rf mkl-dnn-${MKL_DNN_COMMIT}/ && \
    gzip -n mkl-dnn-${MKL_DNN_COMMIT}-patched.tar

# Link examples to workspace
RUN mkdir -p /workspace/nvidia-examples && \
     ln -s /opt/tensorflow/nvidia-examples/* /workspace/nvidia-examples/

ENV CUDA_TOOLKIT_PATH=/usr/local/cuda \
    CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu \
    NCCL_INSTALL_PATH=/usr/lib/x86_64-linux-gnu \
    NCCL_HDR_PATH=/usr/include

# Build and install TF
RUN ./nvbuild.sh --testlist --python$PYVER

# Estimator is installed from pip as a TF dependency
#RUN git clone https://github.com/tensorflow/estimator -b r1.13 /opt/estimator && \
#    cd /opt/estimator && \
#    ln -fs /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
#    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs && \
#    bazel build //tensorflow_estimator/tools/pip_package:build_pip_package && \
#    bazel-bin/tensorflow_estimator/tools/pip_package/build_pip_package /tmp/estimator_pip && \
#    pip install --no-cache-dir --upgrade /tmp/estimator_pip/*.whl && \
#    rm -rf ${HOME}/.cache/bazel /tmp/estimator_pip /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
#    bazel clean --expunge

# Install DALI and build TF plugin, we need to have TF present already
ENV DALI_VERSION=0.8.0 \
    DALI_BUILD=680097
RUN pip install --no-cache-dir --upgrade \
                --extra-index-url https://developer.download.nvidia.com/compute/redist \
                --extra-index-url http://sqrl/dldata/pip-simple --trusted-host sqrl \
        nvidia-dali==${DALI_VERSION} \
        nvidia-dali-tf-plugin==${DALI_VERSION}

ENV TF_ADJUST_HUE_FUSED=1 \
    TF_ADJUST_SATURATION_FUSED=1 \
    TF_ENABLE_WINOGRAD_NONFUSED=1 \
    TF_AUTOTUNE_THRESHOLD=2 \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/tensorflow

# TensorBoard
EXPOSE 6006

# Horovod with fp16 patch
RUN export HOROVOD_GPU_ALLREDUCE=NCCL \
 && export HOROVOD_NCCL_INCLUDE=/usr/include \
 && export HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu \
 && export HOROVOD_NCCL_LINK=SHARED \
 && export HOROVOD_WITHOUT_PYTORCH=1 \
 && export HOROVOD_WITHOUT_MXNET=1 \
 && cd /opt/tensorflow/third_party/horovod \
 && ln -s /usr/local/cuda/lib64/stubs/libcuda.so ./libcuda.so.1 \
 && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD \
 && python setup.py install \
 && python setup.py clean \
 && rm ./libcuda.so.1

# OpenSeq2Seq CTC Decoder & KenLM
RUN patch -p0 < openseq2seq.patch && \
    cd /opt/tensorflow/nvidia-examples/OpenSeq2Seq && \
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
    rm .tf_configure.bazelrc /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    rm -rf ${HOME}/.cache/bazel /tmp/*

# NCF
RUN cd /opt/tensorflow/nvidia-examples/NCF && pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace
COPY NVREADME.md README.md
COPY docker-examples docker-examples
RUN chmod -R a+w /workspace

# Extra defensive wiring for CUDA Compat lib
RUN ln -sf ${_CUDA_COMPAT_PATH}/lib.real ${_CUDA_COMPAT_PATH}/lib \
 && echo ${_CUDA_COMPAT_PATH}/lib > /etc/ld.so.conf.d/00-cuda-compat.conf \
 && ldconfig \
 && rm -f ${_CUDA_COMPAT_PATH}/lib

COPY nvidia_entrypoint.sh /usr/local/bin
ENTRYPOINT ["/usr/local/bin/nvidia_entrypoint.sh"]

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID ${NVIDIA_BUILD_ID:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
ARG NVIDIA_BUILD_REF
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"

