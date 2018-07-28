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
  cd ../../
  PYVER=$(python -c 'import sys; print("{}.{}".format(sys.version_info[0], sys.version_info[1]))')
  ./nvbuild.sh --configonly --python$PYVER
  NUM_GPUS=`nvidia-smi -L | wc -l`

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
fi #aarch64 check

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
FAILS=0

# Note: //tensorflow/python/debug:debugger_cli_common_test fails when run as root due to a file permissions issue.
# Note: //tensorflow/contrib/tensor_forest:scatter_add_ndim_op_test fails for an unknown reason with "Create kernel failed: Invalid argument: AttrValue must not have reference type value of float_ref".
# Note: //tensorflow/contrib/distributions:mvn_full_covariance_test fails due to assert_equal being used to check symmetry of the result of a matmul.
# Note: //tensorflow/contrib/kfac/examples/tests:convnet_test times out when distributed tests are included. These have been commented out in the python test.
# Note: cluster_function_library_runtime_test fails intermitently when run in
#       with 'status: Unavailable: Endpoint read failed'
#       or 'UnknownError: Could not start gRPC server
bazel test --config=cuda -c opt --verbose_failures --local_test_jobs=$NUM_GPUS \
              --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute \
              --test_tag_filters=-no_gpu,-benchmark-test --cache_test_results=no \
              --build_tests_only \
              -- \
              //tensorflow/... \
              //tensorflow/contrib/tensorrt/... \
              //tensorflow/contrib/cudnn_rnn:cudnn_rnn_ops_test \
              `# These are tested in serial below` \
              -//tensorflow/python:localhost_cluster_performance_test \
              -//tensorflow/core/debug:grpc_session_debug_test \
              -//tensorflow/contrib/kfac/examples/tests:convnet_test \
              -//tensorflow/python/kernel_tests:depthtospace_op_test \
              `# We do not provide Go support` \
              -//tensorflow/go/... \
              `# tflite tests fail badly` \
              -//tensorflow/contrib/lite/... \
              `# This is tested by L1_self_test_xla` \
              -//tensorflow/compiler/... \
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
| tee testresult.tmp

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
              //tensorflow/contrib/kfac/examples/tests:convnet_test \
              //tensorflow/python/kernel_tests:depthtospace_op_test \
              //tensorflow/python/kernel_tests:conv_ops_test \
              //tensorflow/core/platform/cloud:ram_file_block_cache_test \
  | tee -a testresult.tmp

FAILS=$((FAILS+$?))

set -e
{ grep "test\.log" testresult.tmp || true; } | ../show_testlogs

exit $FAILS
