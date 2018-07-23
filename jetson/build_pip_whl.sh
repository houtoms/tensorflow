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
JETSON_SWAPFILE=${JETSON_SWAPFILE:-/tf_swapfile}
if [ -f $JETSON_SWAPFILE ]; then
  echo "Swap File already exists.  Enabling now."
  if swapon --show | grep $JETSON_SWAPFILE 2>/dev/null; then
    echo "Swapfile already enabled."
  else
    sudo swapon $JETSON_SWAPFILE
    echo "Swapfile now enabled."
  fi
else
  sudo fallocate -l 4G $JETSON_SWAPFILE
  sudo chmod 600 $JETSON_SWAPFILE
  sudo mkswap $JETSON_SWAPFILE
  sudo swapon $JETSON_SWAPFILE
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR

TF_PYVER=${TF_PYVER:-"2.7"}

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
if [ $TF_PYVER == "2.7" ];  then
  sudo -H pip install $FINAL_WHL_BUILD_PATH/*
else 
  if [ $TF_PYVER != "3.5" ]; then
    echo "Python version must be either 2.7 or 3.5.  Exiting now."
    exit 1
  else
    sudo -H pip3 install $FINAL_WHL_BUILD_PATH/*
  fi
fi
TF_BUILD_TIME=$SECONDS

#Remove Swapfile 
sudo swapoff $JETSON_SWAPFILE
sudo rm $JETSON_SWAPFILE

#Set NvpModel mode back to ARM cores only
sudo nvpmodel -m 3

#Print Total Build Time
echo "Tensorflow Build Time: $TF_BUILD_TIME"


