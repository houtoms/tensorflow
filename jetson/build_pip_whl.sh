#!/bin/bash
# Build Tensorflow from fresh jetson
#
#	1) Check for and install dependencies
#	2) Run the configure script
#	3) Compile tensorflow with bazel
#	4) Build the pip whl
#	5) Install the tensorflow package with pip

set -o pipefail
set -e

#Enable Denver cores for faster building
sudo nvpmodel -m 0

#Create a swap file for use in tensorflow compilation
sudo fallocate -l 4G /tf_swapfile
sudo chmod 600 /tf_swapfile
sudo mkswap /tf_swapfile
sudo swapon /tf_swapfile

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR

SECONDS=0
#Install required dependencies
sudo -H bash check_deps.sh

#Set configuration options and run configure script
sudo -H bash auto_conf.sh

#Compile and link tensorflow with bazel
bazel build --config=opt --config=cuda ../tensorflow/tools/pip_package/build_pip_package

#Build the pip whl from the main Tensorflow directory
FINAL_WHL_BUILD_PATH=${FINAL_WHL_BUILD_PATH:-/tmp/tensorflow_pkg}
cd ../
bash tensorflow/tools/pip_package/build_pip_package.sh $FINAL_WHL_BUILD_PATH 
cd jetson/

#Install the Tensorflow package
sudo -H pip install $FINAL_WHL_BUILD_PATH/*
TF_BUILD_TIME=$SECONDS

#Remove Swapfile 
sudo swapoff /tf_swapfile
sudo rm /tf_swapfile

#Set NvpModel mode back to ARM cores only
sudo nvpmodel -m 3

#Print Total Build Time
echo $TF_BUILD_TIME


