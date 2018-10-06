#!/bin/bash

# This script should be called after sourcing setup_env.sh to configure
# the environment variables. This script performs the following steps. 
# 
# 1. Installs necessary debian/python package dependencies
# 2. Installs the tensorflow/models repository for object detection
# 3. Installs the pycoco API

sudo apt install python-pip python-pil python-matplotlib python-tk
sudo apt install python3-pip python3-pil python3-matplotlib python3-tk

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

python setup.py install --user

popd

pushd $SLIM_DIR
python setup.py install --user
popd

# INSTALL PYCOCOTOOLS

pushd $PYCOCO_DIR
python setup.py install --user
popd
