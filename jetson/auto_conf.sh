#!/bin/bash

set -o pipefail
set -e

#This script sets Tensorflow's configuration options and runs the configure script


TF_PYPATH=`which python`

export PYTHON_BIN_PATH="$TF_PYPATH"
export TF_NEED_S3="0"
export TF_NEED_OPENCL_SYCL="0"
export TF_NEED_CUDA="1"

export CUDA_TOOLKIT_PATH="/usr/local/cuda"
export TF_CUDA_VERSION="9.0"

export CUDNN_INSTALL_PATH="${CUDA_TOOLKIT_PATH}-${TF_CUDA_VERSION}"

export TF_CUDNN_VERSION="7"
export TF_NEED_TENSORRT="1"
export TENSORRT_INSTALL_PATH="/usr/lib/aarch64-linux-gnu"
export TF_TENSORRT_VERSION="4.0.4"
export TF_NCCL_VERSION="1"
export TF_CUDA_COMPUTE_CAPABILITIES="6.2"
export TF_CUDA_CLANG="0"
export GCC_HOST_COMPILER_PATH="/usr/bin/gcc"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "" | ${SCRIPT_DIR}/../configure
