#!/bin/bash

set -o pipefail
set -e

#This script sets Tensorflow's configuration options and runs the configure script


TF_PYPATH=`which python`

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export PYTHON_BIN_PATH="$TF_PYPATH"
export TF_NEED_AWS="0"
export TF_NEED_OPENCL_SYCL="0"
export TF_NEED_CUDA="1"

export CUDA_TOOLKIT_PATH="/usr/local/cuda"

export TF_CUDA_VERSION=$(dpkg -l |grep cuda-cudart | awk '{print $3}' | cut -d . -f 1-2 | sort -u)
export TF_CUDNN_VERSION=$(dpkg -l |grep libcudnn | awk '{print $3}' | cut -d . -f 1-2 | sort -u)
export TF_TENSORRT_VERSION=$(dpkg -l |grep tensorrt | awk '{print $3}' | cut -d . -f 1-3 | sort -u)
case "${TF_CUDA_VERSION}" in
  9.*)  export TF_CUDA_COMPUTE_CAPABILITIES="5.3,6.2" ;;
  10.*) export TF_CUDA_COMPUTE_CAPABILITIES="5.3,6.2,7.2" ;;
esac

export CUDNN_INSTALL_PATH="${CUDA_TOOLKIT_PATH}-${TF_CUDA_VERSION}"

export TF_NEED_TENSORRT="1"
export TENSORRT_INSTALL_PATH="/usr/lib/aarch64-linux-gnu"
export TF_NCCL_VERSION="1"
export TF_CUDA_CLANG="0"
export GCC_HOST_COMPILER_PATH="/usr/bin/gcc"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "" | ${SCRIPT_DIR}/../configure
