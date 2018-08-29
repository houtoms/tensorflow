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

NATIVE_ARCH=`uname -m`
if [ ${NATIVE_ARCH} == 'aarch64' ]; then
  NUM_GPUS=1
  bash ./jetson/auto_conf.sh
else
  PYVER=$(python -c 'import sys; print("{}.{}".format(sys.version_info[0], sys.version_info[1]))')
  ./nvbuild.sh --configonly --python$PYVER
  NUM_GPUS=`nvidia-smi -L | wc -l` 

  echo "Installing test dependencies..."
  CHECK tensorflow/tools/ci_build/install/install_bootstrap_deb_packages.sh
  CHECK tensorflow/tools/ci_build/install/install_deb_packages.sh
  CHECK add-apt-repository -y ppa:openjdk-r/ppa
  CHECK add-apt-repository -y ppa:george-edison55/cmake-3.x
  if [[ "${PYVER%.*}" == "3" ]]; then
    CHECK tensorflow/tools/ci_build/install/install_python${PYVER}_pip_packages.sh
  else
    CHECK tensorflow/tools/ci_build/install/install_pip_packages.sh
  fi
  CHECK tensorflow/tools/ci_build/install/install_proto3.sh
  CHECK tensorflow/tools/ci_build/install/install_auditwheel.sh
fi

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

echo "Building and running tests..."
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
2>&1 | tee testresult.tmp | grep '^\[\|^FAIL\|^Executed\|Build completed'
RESULT=$?

set -e
{ grep "test\.log" testresult.tmp || true; } | ./qa/show_testlogs

exit $RESULT
