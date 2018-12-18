#!/bin/bash

set -e
set -o pipefail
set +x

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
  ./nvbuild.sh --configonly --python$PYVER

  #echo "Installing test dependencies..."
  #CHECK tensorflow/tools/ci_build/install/install_bootstrap_deb_packages.sh
  #CHECK tensorflow/tools/ci_build/install/install_deb_packages.sh
  #CHECK add-apt-repository -y ppa:openjdk-r/ppa
  #CHECK add-apt-repository -y ppa:george-edison55/cmake-3.x
  #if [[ "${PYVER%.*}" == "3" ]]; then
  #  CHECK tensorflow/tools/ci_build/install/install_python${PYVER}_pip_packages.sh
  #else
  #  CHECK tensorflow/tools/ci_build/install/install_pip_packages.sh
  #fi
  #CHECK tensorflow/tools/ci_build/install/install_proto3.sh
  #CHECK tensorflow/tools/ci_build/install/install_auditwheel.sh
fi #aarch64 check

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

echo "Building and running tests..."
set +e
FAILS=0

# Note: //tensorflow/python/debug:debugger_cli_common_test fails when run as root due to a file permissions issue.
# Note: //tensorflow/contrib/tensor_forest:scatter_add_ndim_op_test fails for an unknown reason with "Create kernel failed: Invalid argument: AttrValue must not have reference type value of float_ref".
# Note: //tensorflow/contrib/distributions:mvn_full_covariance_test fails due to assert_equal being used to check symmetry of the result of a matmul.
# Note: cluster_function_library_runtime_test fails intermitently when run in
#       with 'status: Unavailable: Endpoint read failed'
#       or 'UnknownError: Could not start gRPC server
# TODO(benbarsdell): Re-enable local_client_execute_test_gpu once CUDA version is > 9.0
#                      (due to known issue with PTX generation).
bazel test --config=cuda -c opt --verbose_failures --local_test_jobs=$GPUS \
              --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute \
              --test_tag_filters=-no_gpu,-benchmark-test --cache_test_results=no \
              --build_tests_only \
              -- \
              //tensorflow/... \
              //tensorflow/contrib/tensorrt/... \
              `# These are tested in serial below` \
              -//tensorflow/python:localhost_cluster_performance_test \
              -//tensorflow/core/debug:grpc_session_debug_test \
              -//tensorflow/python/kernel_tests:depthtospace_op_test \
              `# We do not provide Go support` \
              -//tensorflow/go/... \
              `# tflite tests fail badly` \
              -//tensorflow/contrib/lite/... \
              `# grpc_session_test_gpu is a flaker.`\
              `# It usually runs in 370 sec, but sumtimes timesout after 900.` \
              -//tensorflow/core/distributed_runtime/rpc:grpc_session_test_gpu \
              `# Minor failures` \
              -//tensorflow/python/kernel_tests:atrous_conv2d_test \
              -//tensorflow/python/debug:debugger_cli_common_test \
              -//tensorflow/python/eager:core_test \
              -//tensorflow/contrib/tensor_forest:scatter_add_ndim_op_test \
              -//tensorflow/contrib/distributions:mvn_full_covariance_test \
              -//tensorflow/contrib/factorization:gmm_test \
              -//tensorflow/core/distributed_runtime:cluster_function_library_runtime_test \
              `# These are flakers as of tf 1.7. Keeping ram_file test below.` \
              -//tensorflow/contrib/learn:monitors_test \
              -//tensorflow/core/platform/cloud:ram_file_block_cache_test \
              `# conv_ops_test has timed out in some M40 runs. Moved to serial tests below.` \
              -//tensorflow/python/kernel_tests:conv_ops_test \
              `# data_utils_test hangs.` \
              -//tensorflow/python/keras:data_utils_test \
              -//tensorflow/compiler/xla/tests:local_client_execute_test_gpu \
              -//tensorflow/compiler/xla/python:xla_client_test \
              `# biasadd_matmul seems to suffer from but in TRT. [B] 2432079` \
              -//tensorflow/contrib/tensorrt:biasadd_matmul_test \
              -//tensorflow/compiler/xla/tests:exhaustive_f32_elementwise_op_test \
              -//tensorflow/contrib/cudnn_rnn:cudnn_rnn_ops_test \
2>&1 | tee testresult.tmp | grep '^\[\|^FAIL\|^Executed\|Build completed'

FAILS=$((FAILS+$?))


# Note: The first two tests were observed to fail intermittently with error
#       "address already in use" when run as part of the above command
#       on a DGX-1. The others timed out in some runs.
bazel test    --config=cuda -c opt --verbose_failures --local_test_jobs=1 \
              --test_tag_filters=-no_gpu,-benchmark-test --cache_test_results=no \
              --build_tests_only \
              -- \
              //tensorflow/python:localhost_cluster_performance_test \
              //tensorflow/core/debug:grpc_session_debug_test \
              //tensorflow/python/kernel_tests:depthtospace_op_test \
              //tensorflow/python/kernel_tests:conv_ops_test \
              //tensorflow/core/platform/cloud:ram_file_block_cache_test \
2>&1 | tee -a testresult.tmp | grep '^\[\|^FAIL\|^Executed\|Build completed'

FAILS=$((FAILS+$?))

set -e
{ grep "test\.log" testresult.tmp || true; } | ./qa/show_testlogs

exit $FAILS
