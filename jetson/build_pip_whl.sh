#!/bin/bash

#This script should:
#	1) Run the configure script
#	2) Compile tensorflow with bazel
#	3) Build the pip whl
#	4) Install the tensorflow package with pip

#TODO automatically input the options for the configuration script:
export PYTHON_BIN_PATH="/usr/bin/python"
export PYTHON_LIB_PATH="/usr/local/lib/python2.7/dist-packages"
export TF_NEED_OPENCL_SYCL="0"
export TF_NEED_CUDA="1"
export CUDA_TOOLKIT_PATH="/usr/local/cuda"
export TF_CUDA_VERSION="9.0"
export CUDNN_INSTALL_PATH="/usr/local/cuda-9.0"
export TF_CUDNN_VERSION="7"
export TF_NEED_TENSORRT="1"
export TENSORRT_INSTALL_PATH="/usr/lib/aarch64-linux-gnu"
export TF_TENSORRT_VERSION="4.0.4"
export TF_NCCL_VERSION="1"
export TF_CUDA_COMPUTE_CAPABILITIES="6.2"
export TF_CUDA_CLANG="0"
export GCC_HOST_COMPILER_PATH="/usr/bin/gcc"

#Result="\n\nn\nn\nn\nn\nn\nn\nn\nn\nn\ny\n9\n\n\n\ny\n\n\n6.2\nn\n\nn\n\nn\n"
#echo -e $Result | ../configure

echo "" | ../configure



bazel build --config=opt --config=cuda ../tensorflow/tools/pip_package/build_pip_package
#TODO check result of build command, if the build failed throw error

bash ../tensorflow/tools/pip_package/build_pip_package.sh /tmp/tensorflow_pkg


#TODO check if tensorflow is installed, act based thereon
#sudo -H pip install /tmp/tensorflow_pkg/*






