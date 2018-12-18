#!/bin/bash

# This script should be called after sourcing setup_env.sh to configure
# the environment variables. This script performs the following steps. 
# 
# 1. Installs necessary python package dependencies
# 2. Installs the tensorflow/models repository for object detection
# 3. Installs the pycoco API

# Python 2 matplotlib needs python-tk; do nothing for Py3
python -V 2>&1 | grep "Python 3" || \
  ( export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y --no-install-recommends python-tk )

RESEARCH_DIR=$TF_MODELS_DIR/research
SLIM_DIR=$RESEARCH_DIR/slim
PYCOCO_DIR=$COCO_API_DIR/PythonAPI

pushd $RESEARCH_DIR

# GET PROTOC 3.5

BASE_URL="https://github.com/google/protobuf/releases/download/v3.5.1/"
PROTOC_DIR=protoc
PROTOC_EXE=$PROTOC_DIR/bin/protoc

mkdir -p $PROTOC_DIR
pushd $PROTOC_DIR
ARCH=$(uname -m)
if [ "$ARCH" == "aarch64" ] ; then
  filename="protoc-3.5.1-linux-aarch_64.zip"
elif [ "$ARCH" == "x86_64" ] ; then
  filename="protoc-3.5.1-linux-x86_64.zip"
else
  echo ERROR: $ARCH not supported.
  exit 1;
fi
wget --no-check-certificate ${BASE_URL}${filename}
unzip ${filename}
popd

# BUILD PROTOBUF FILES
$PROTOC_EXE object_detection/protos/*.proto --python_out=.

# INSTALL OBJECT DETECTION

pip install -e .

popd

pushd $SLIM_DIR
pip install -e .
popd

# INSTALL PYCOCOTOOLS

pushd $PYCOCO_DIR
pip install -e .
popd
