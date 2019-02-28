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
python${PYVER} -m virtualenv ./tf_env${PYVER}

# Activate the virtual environment; from here on, python refers to the desired version
source ./tf_env${PYVER}/bin/activate

# Install required Python packages
# TODO: Hardcoded package versions must be kept in sync with the container
pip${PYVER} install numpy==1.16.1 enum34 mock h5py==2.9.0 keras_applications==1.0.6 keras_preprocessing==1.0.5

# Set configuration options and run configure script
source jetson/auto_conf.sh

# Run script without capturing output first to fail if JP version is unknown
bash ${SCRIPT_DIR}/get_jpver.sh
# Determine JetPack version for wheel naming
JPVER=$(${SCRIPT_DIR}/get_jpver.sh)

export OUTPUT_DIRS="wheelhouse/${JPVER}/kernel_tests/ wheelhouse/${JPVER}/xla_tests/ wheelhouse/${JPVER}/"
export IN_CONTAINER="0"
export NOCLEAN="1"
export TESTLIST="1"
export LIBCUDA_FOUND="1"
export BUILD_OPTS="jetson/bazelopts"
export PYVER
bash bazel_build.sh

# Clean up
deactivate

popd
