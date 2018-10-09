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

# Set up virtualenv
#
# The complexity here is that in some of our systems, the virtualenv executable
#  is called 'virtualenv', while on others it's either 'virtualenv2' or 'virtualenv3' only.
#  It turns out it doesn't matter which of them you use as long as you specify --python=...
#  and tell it which python version you want to be placed in the virtual environment.
#
VIRTUALENV=$(which virtualenv 2>/dev/null || which virtualenv2 2>/dev/null || which virtualenv3)
${VIRTUALENV} --python=$(which python${PYVER}) ./tf_env${PYVER}

# Activate the virtual environment; from here on, python refers to the desired version
source ./tf_env${PYVER}/bin/activate

# Install required Python packages
pip${PYVER} install numpy enum34 mock

# Set configuration options and run configure script
source jetson/auto_conf.sh

# Determine JetPack version for wheel naming
JPVER=$(${SCRIPT_DIR}/get_jpver.sh)

# Compile and link tensorflow with bazel, package wheel
time (
bazel build --config=opt --config=cuda tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package ./wheelhouse/${JPVER}/ --gpu
#bazel clean --expunge
#rm -rf ${HOME}/.cache/bazel
)

# Clean up
deactivate

popd
