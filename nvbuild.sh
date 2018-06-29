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
}

PYVER=2.7
CONFIGONLY=0
NOCONFIG=0
NOCLEAN=0

while [[ $# -gt 0 ]]; do
  case $1 in
    "--help"|"-h")  Usage; exit 1 ;;
    "--python2.7")  PYVER=2.7 ;;
    "--python3.5")  PYVER=3.5 ;;
    "--configonly") CONFIGONLY=1 ;;
    "--noconfig")   NOCONFIG=1 ;;
    "--noclean")    NOCLEAN=1 ;;
    *)
      echo UNKNOWN OPTION $1
      echo Run $0 -h for help
      exit 1
  esac
  shift 1
done

cd /opt/tensorflow
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

bazel build -c opt --config=cuda tensorflow/tools/pip_package:build_pip_package --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip --gpu
pip$PYVER install --no-cache-dir --upgrade /tmp/pip/tensorflow_gpu-*.whl
rm -f /tmp/pip/tensorflow_gpu-*.whl
if [[ $NOCLEAN -eq 0 ]]; then
  bazel clean --expunge
  rm -rf /root/.cache/bazel
  rm .tf_configure.bazelrc .bazelrc
  if [[ "$LIBCUA_FOUND" -eq 0 ]]; then
    rm /usr/local/cuda/lib64/stubs/libcuda.so.1
  fi
fi
