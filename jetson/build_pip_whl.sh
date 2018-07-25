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


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR


SECONDS=0
#Install required dependencies

TF_PYVER=${TF_PYVER:-"2"}


#Install virtualenv
python${TF_PYVER} -m virtualenv tf_env

#Activate the virtual environment; from here on, python refers to the desired version
source tf_env/bin/activate

#Install bazel
mkdir -p ~/Bazel
pushd ~/Bazel
wget https://github.com/bazelbuild/bazel/releases/download/0.13.0/bazel-0.13.0-dist.zip
unzip bazel-0.13.0-dist.zip
bash compile.sh
popd


#Install required pip packages
pip install -y numpy enum34 mock


#Set configuration options and run configure script
bash auto_conf.sh

#Compile and link tensorflow with bazel
bazel build --config=opt --config=cuda ../tensorflow/tools/pip_package/build_pip_package

#Build the pip whl from the main Tensorflow directory
FINAL_WHL_BUILD_PATH=${FINAL_WHL_BUILD_PATH:-/tmp/tensorflow_pkg}
cd ../
bash tensorflow/tools/pip_package/build_pip_package.sh $FINAL_WHL_BUILD_PATH 
cd jetson/

#Install the Tensorflow package (user is within virtual env)
pip install -y $FINAL_WHL_BUILD_PATH/*

TF_BUILD_TIME=$SECONDS

#Print Total Build Time
echo "Tensorflow Build Time: $TF_BUILD_TIME Seconds"


