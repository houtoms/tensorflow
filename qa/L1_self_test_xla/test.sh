#!/bin/bash

set -e
set -o pipefail

NATIVE_ARCH=`uname -m`
if [ ${NATIVE_ARCH} == 'aarch64' ]; then
  NUM_GPUS=1
  pushd ../../jetson
  bash auto_conf.sh
  popd
else
  pushd ../../
  PYVER=$(python -c 'import sys; print("{}.{}".format(sys.version_info[0], sys.version_info[1]))')
  ./nvbuild.sh --configonly --python$PYVER
  NUM_GPUS=`nvidia-smi -L | wc -l` 
  popd
fi

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

set +e

# TODO(benbarsdell): Re-enable local_client_execute_test_gpu once CUDA version is > 9.0
#                      (due to known issue with PTX generation).
bazel test --config=cuda -c opt --verbose_failures --local_test_jobs=$NUM_GPUS \
              --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute \
              --test_tag_filters=-no_gpu,-benchmark-test \
              --build_tests_only \
              -- \
              //tensorflow/compiler/... \
              -//tensorflow/compiler/xla/tests:local_client_execute_test_gpu \
              -//tensorflow/compiler/xla/python:xla_client_test \
| tee testresult.tmp
RESULT=$?

set -e
{ grep "test\.log" testresult.tmp || true; } | ../show_testlogs

exit $RESULT
