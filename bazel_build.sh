#!/bin/bash
# Build the components of tensorflow that require Bazel


# Inputs: 
# 	OUTPUT_DIRS - String of space-delimited directories to store outputs, in order of:
#			1)kernel test list 
#			2)xla test list 
#			3)tensorflow whl
#	TESTLIST - Determines whether the test lists are built (1 to build, 0 to skip)
#	NOCLEAN - Determines whether bazel clean is run and the tensorflow whl is 
#			removed after the build and install (0 to clean, 1 to skip)
#	PYVER - The version of python
#	BUILD_OPTS - File containing desired bazel flags for building tensorflow
#	LIBCUDA_FOUND - Determines whether a libcuda stub was created and needs to be cleaned (0 to clean, 1 to skip)
#	IN_CONTAINER - Flag for whether Tensorflow is being built within a container (1 for yes, 0 for bare-metal)
#

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"


read -ra OUTPUT_LIST <<<"$OUTPUT_DIRS"
KERNEL_OUT=${OUTPUT_LIST[0]}
XLA_OUT=${OUTPUT_LIST[1]}
WHL_OUT=${OUTPUT_LIST[2]}


# Build the test lists for L1 kernel and xla tests
if [[ $TESTLIST -eq 1 ]]; then
  rm -f "tests.list" \
        "tensorflow/compiler/tests/tests.list"

  bazel test --config=cuda -c opt --verbose_failures --local_test_jobs=1 \
             --run_under="$THIS_DIR/tools/test_grabber.sh $KERNEL_OUT" \
             --build_tests_only --test_tag_filters=-no_gpu,-benchmark-test \
             --cache_test_results=no -- \
             //tensorflow/python/kernel_tests/... \
             `# The following tests are skipped becaues they depend on additional binaries.` \
             -//tensorflow/python/kernel_tests:ackermann_test \
             -//tensorflow/python/kernel_tests:duplicate_op_test \
             -//tensorflow/python/kernel_tests:invalid_op_test
  bazel test --config=cuda -c opt --verbose_failures --local_test_jobs=1 \
             --run_under="$THIS_DIR/tools/test_grabber.sh $XLA_OUT" \
             --build_tests_only --test_tag_filters=-no_gpu,-benchmark-test \
             --cache_test_results=no -- \
             //tensorflow/compiler/tests/... \
             `# The following tests are skipped becaues they depend on additional binaries.` \
             -//tensorflow/compiler/tests:reduce_window_test \
             -//tensorflow/compiler/tests:while_test \
             -//tensorflow/compiler/tests:dynamic_slice_ops_test \
             -//tensorflow/compiler/tests:while_test \
             -//tensorflow/compiler/tests:sort_ops_test \
             -//tensorflow/compiler/tests:reduce_window_test \
             -//tensorflow/compiler/tests:dynamic_slice_ops_test \
             -//tensorflow/compiler/tests:sort_ops_test
fi

if [[ $IN_CONTAINER -eq 1 ]]; then
  bazel build $(cat $BUILD_OPTS) \
      tensorflow/tools/pip_package:build_pip_package \
      //tensorflow:libtensorflow_cc.so \
      //tensorflow:libtensorflow_framework.so
  mkdir -p /usr/local/lib/tensorflow
  cp bazel-bin/tensorflow/libtensorflow_*.so /usr/local/lib/tensorflow/
else
  bazel build $(cat $BUILD_OPTS) \
      tensorflow/tools/pip_package:build_pip_package
fi
bazel-bin/tensorflow/tools/pip_package/build_pip_package $WHL_OUT --gpu
pip$PYVER install --no-cache-dir --upgrade $WHL_OUT/tensorflow_gpu-*.whl
if [[ $NOCLEAN -eq 0 ]]; then
  rm -f $WHL_OUT/tensorflow_gpu-*.whl
  bazel clean --expunge
  rm .tf_configure.bazelrc .bazelrc
  rm -rf ${HOME}/.cache/bazel
  if [[ "$LIBCUDA_FOUND" -eq 0 ]]; then
    rm /usr/local/cuda/lib64/stubs/libcuda.so.1
  fi
fi


