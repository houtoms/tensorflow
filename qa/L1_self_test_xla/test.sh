#!/bin/bash

set -e
set -o pipefail

nvidia-smi
cd /opt/tensorflow
PYVER=$(python -c 'import sys; print("{}.{}".format(sys.version_info[0], sys.version_info[1]))')
./nvbuild.sh --configonly --python$PYVER

tensorflow/tools/ci_build/install/install_bootstrap_deb_packages.sh
tensorflow/tools/ci_build/install/install_deb_packages.sh
add-apt-repository -y ppa:openjdk-r/ppa && \
  add-apt-repository -y ppa:george-edison55/cmake-3.x

if [[ "${PYVER%.*}" == "3" ]]; then
  tensorflow/tools/ci_build/install/install_python${PYVER}_pip_packages.sh
else
  tensorflow/tools/ci_build/install/install_pip_packages.sh
fi
tensorflow/tools/ci_build/install/install_proto3.sh
tensorflow/tools/ci_build/install/install_auditwheel.sh

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

set +e

# TODO(benbarsdell): Re-enable local_client_execute_test_gpu once CUDA version is > 9.0
#                      (due to known issue with PTX generation).
NUM_GPUS=`nvidia-smi -L | wc -l` && \
  bazel test  --config=cuda -c opt --verbose_failures --local_test_jobs=$NUM_GPUS \
              --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute \
              --test_tag_filters=-no_gpu,-benchmark-test \
              --build_tests_only \
              -- \
              //tensorflow/compiler/... \
              -//tensorflow/compiler/xla/tests:local_client_execute_test_gpu \
  | tee testresult.tmp && { grep "test\.log" testresult.tmp || true; } \
  | /opt/tensorflow/qa/show_testlogs

exit $?
