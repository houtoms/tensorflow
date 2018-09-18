#!/bin/bash
#
# Configure, build, and install Tensorflow
#

# Exit at error
set -e

Usage() {
  echo "Configure, build, and install Tensorflow."
  echo ""
  echo "  Usage: $0 [OPTIONS]"
  echo ""
  echo "    OPTIONS          DESCRIPTION"
  echo "    --python2.7      Build python2.7 package (default)"
  echo "    --python3.5      Build python3.5 package"
  echo "    --configonly     Run configure step only"
  echo "    --noconfig       Skip configure step"
  echo "    --noclean        Retain intermediate build files"
  echo "    --testlist       Build list of python kernel_tests"
}

PYVER=2.7
CONFIGONLY=0
NOCONFIG=0
NOCLEAN=0
TESTLIST=0

while [[ $# -gt 0 ]]; do
  case $1 in
    "--help"|"-h")  Usage; exit 1 ;;
    "--python2.7")  PYVER=2.7 ;;
    "--python3.5")  PYVER=3.5 ;;
    "--configonly") CONFIGONLY=1 ;;
    "--noconfig")   NOCONFIG=1 ;;
    "--noclean")    NOCLEAN=1 ;;
    "--testlist")   TESTLIST=1 ;;
    *)
      echo UNKNOWN OPTION $1
      echo Run $0 -h for help
      exit 1
  esac
  shift 1
done

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

export TF_CUDA_VERSION=$(echo "${CUDA_VERSION}" | cut -d . -f 1-2)
export TF_CUDNN_VERSION=$(echo "${CUDNN_VERSION}" | cut -d . -f 1)
export TF_NEED_CUDA=1
export TF_CUDA_COMPUTE_CAPABILITIES="5.2,6.0,6.1,7.0,7.5"
export TF_NEED_HDFS=0
export TF_ENABLE_XLA=1
export TF_NEED_TENSORRT=1
export TF_NCCL_VERSION=2
export CC_OPT_FLAGS="-march=sandybridge -mtune=broadwell"

cd "$THIS_DIR"
export PYTHON_BIN_PATH=/usr/bin/python$PYVER
LIBCUDA_FOUND=$(ldconfig -p | awk '{print $1}' | grep libcuda.so | wc -l)
if [[ $NOCONFIG -eq 0 ]]; then
  if [[ "$LIBCUDA_FOUND" -eq 0 ]]; then
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs
      ln -fs /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
  fi
  yes "" | ./configure
fi

if [[ $CONFIGONLY -eq 1 ]]; then
  exit 0
fi

if [[ $TESTLIST -eq 1 ]]; then
  rm -f "tensorflow/python/kernel_tests/tests.list" \
        "tensorflow/compiler/tests/test.list"
  
  bazel test --config=cuda -c opt --verbose_failures --local_test_jobs=1 \
             --run_under="$THIS_DIR/tools/test_grabber.sh tensorflow/python/kernel_tests" \
             --build_tests_only --test_tag_filters=-no_gpu,-benchmark-test \
             --cache_test_results=no -- \
             //tensorflow/python/kernel_tests/... \
             `# The following tests are skipped becaues they depend on additional binaries.` \
             -//tensorflow/python/kernel_tests:ackermann_test \
             -//tensorflow/python/kernel_tests:duplicate_op_test \
             -//tensorflow/python/kernel_tests:invalid_op_test
  bazel test --config=cuda -c opt --verbose_failures --local_test_jobs=1 \
             --run_under="$THIS_DIR/tools/test_grabber.sh tensorflow/compiler/tests" \
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

bazel build -c opt --config=cuda tensorflow/tools/pip_package:build_pip_package --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip --gpu
pip$PYVER install --no-cache-dir --upgrade /tmp/pip/tensorflow_gpu-*.whl
rm -f /tmp/pip/tensorflow_gpu-*.whl
if [[ $NOCLEAN -eq 0 ]]; then
  bazel clean --expunge
  rm -rf /root/.cache/bazel
  rm .tf_configure.bazelrc .bazelrc
  if [[ "$LIBCUDA_FOUND" -eq 0 ]]; then
    rm /usr/local/cuda/lib64/stubs/libcuda.so.1
  fi
fi
