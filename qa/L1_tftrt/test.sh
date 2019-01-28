#!/bin/bash

set -e
set -o pipefail

CHECK_TMP=$(mktemp)
trap "/bin/rm -f $CHECK_TMP" EXIT

function CHECK {
    ${1+"$@"} > "$CHECK_TMP" 2>&1 && RC=0 || RC=1
    if [[ $RC -eq 1 ]]; then
        echo "'${1+"$@"}'" exited with error. ABORTING
        cat "$CHECK_TMP"
    fi
    return $RC
}

cd ../..

export TEST_TMPDIR="${HOME}/.cache/bazel-py${PYVER}/"

GPUS=$(nvidia-smi -L 2>/dev/null| wc -l || echo 1)

NATIVE_ARCH=`uname -m`
if [ ${NATIVE_ARCH} == 'aarch64' ]; then
  bash ./jetson/auto_conf.sh
else
  PYVER=$(python -c 'import sys; print("{}.{}".format(sys.version_info[0], sys.version_info[1]))')
  CHECK ./nvbuild.sh --configonly --python$PYVER

  CHECK apt-get update
  CHECK apt-get install -y openjdk-8-jdk
fi #aarch64 check


echo "Building and running tests..."
set +e

bazel test --config=cuda -c opt --verbose_failures --local_test_jobs=$GPUS \
              --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute \
              --test_tag_filters=-no_gpu,-benchmark-test --cache_test_results=no \
              --build_tests_only \
              -- \
              //tensorflow/contrib/tensorrt/... \
2>&1 | tee testresult.tmp | grep '^\[\|^FAIL\|^Executed\|Build completed'

FAILS=$?

set -e
{ grep "test\.log" testresult.tmp || true; } | ./qa/show_testlogs

exit $FAILS
