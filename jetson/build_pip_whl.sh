#!/bin/bash
# Build Tensorflow from fresh jetson
#
#	1) Check for and install dependencies
#	2) Run the configure script to set env vars
#	3) Compile tensorflow with bazel
#	4) Build the pip whl
#	5) Install the tensorflow package with pip

set -o pipefail
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pushd ${SCRIPT_DIR}/..

PYVER=${PYVER:-"2.7"}

# Set up virtualenv --- TODO: should this be in the before_script?
python${PYVER} -m virtualenv ./tf_env

# Activate the virtual environment; from here on, python refers to the desired version
source ./tf_env/bin/activate

# Install required Python packages
pip${PYVER} install numpy enum34 mock

# Set configuration options and run configure script
source jetson/auto_conf.sh

# Compile and link tensorflow with bazel, package wheel
time (
bazel build --config=opt --config=cuda tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package ./wheelhouse/ --gpu
#bazel clean --expunge
#rm -rf ${HOME}/.cache/bazel
)

# Clean up
deactivate
#rm -rf ./tf_env

popd
