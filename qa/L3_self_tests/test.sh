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
  CHECK ./nvbuild.sh --configonly --python$PYVER

  CHECK apt-get update
  CHECK apt-get install -y openjdk-8-jdk
fi #aarch64 check

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

echo "Building and running tests..."
set +e

bazel test --config=cuda -c opt --verbose_failures --local_test_jobs=$GPUS \
              --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute \
              --test_tag_filters=-no_gpu,-benchmark-test --cache_test_results=no \
              --build_tests_only \
              -- \
              //tensorflow/core/... \
              //tensorflow/compiler/... \
              //tensorflow/stream_executor/... \
              //tensorflow/python/... \
              //tensorflow/contrib/... \
              `# data_utils_test known to hang` \
              -//tensorflow/python/keras:data_utils_test \
              `# debugger_cli_common_test fails when run as root due to filer permission issue` \
              -//tensorflow/python/debug:debugger_cli_common_test \
              `# grpc_session_test_gpu is a flaker.`\
              `# It usually runs in 370 sec, but sumtimes timesout after 900.` \
              -//tensorflow/core/distributed_runtime/rpc:grpc_session_test_gpu \
              `# Monitors_test is a flaker as of tf 1.7.` \
              -//tensorflow/contrib/learn:monitors_test \
              `# Minor failure` \
              -//tensorflow/contrib/factorization:gmm_test \
              `# cluster_function_library_runtime_test fails intermitently with Endpoint read failed or Could not start gRPC server` \
              -//tensorflow/core/distributed_runtime:cluster_function_library_runtime_test \
              `# resampler tests fail to build in r1.13` \
              -//tensorflow/contrib/resampler/... \
              `# f32 elementwise tests seem to assume TF compiled without GPU support` \
              -//tensorflow/compiler/xla/tests:exhaustive_f32_elementwise_op_test_cpu \
              `# Tests fail when executed eagerly because optimizer is not instance of tf.train.Optimier. ` \
              -//tensorflow/python/keras:training_generator_test \
              `# Check failed: IsAligned()` \
              -//tensorflow/python/data/experimental/kernel_tests/optimization:map_vectorization_test \
              `# As of 1.13.0-rc0 distribute mirrored multi-gpu fails with IndexError: pop from empty list` \
              -//tensorflow/contrib/distribute/python:mirrored_strategy_multigpu_test \
              -//tensorflow/contrib/distribute/python:mirrored_strategy_multigpu_test_gpu \
2>&1 | tee testresult.tmp | grep '^\[\|^FAIL\|^Executed\|Build completed'

FAILS=$?

set -e
{ grep "test\.log" testresult.tmp || true; } | ./qa/show_testlogs

exit $FAILS
