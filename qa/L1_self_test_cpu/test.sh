#!/bin/bash

nvidia-smi
cd /opt/tensorflow
curl -O https://bootstrap.pypa.io/get-pip.py && \
  python get-pip.py && \
  python3 get-pip.py && \
  pip2 install --upgrade --force-reinstall pip && \
  rm get-pip.py

tensorflow/tools/ci_build/install/install_bootstrap_deb_packages.sh
add-apt-repository -y ppa:openjdk-r/ppa && \
  add-apt-repository -y ppa:george-edison55/cmake-3.x
tensorflow/tools/ci_build/install/install_deb_packages.sh

pip  install wheel six==1.10.0 protobuf==3.1.0 numpy==1.11.0 scipy==0.16.1 scikit-learn==0.17.1 pandas==0.18.1 psutil py-cpuinfo pylint pep8 portpicker mock

pip3 install wheel six==1.10.0 protobuf==3.1.0 numpy==1.11.0 scipy==0.16.1 scikit-learn==0.17.1 pandas==0.18.1 psutil py-cpuinfo pylint pep8 portpicker

tensorflow/tools/ci_build/install/install_pip_packages.sh
tensorflow/tools/ci_build/install/install_proto3.sh
tensorflow/tools/ci_build/install/install_auditwheel.sh

pip uninstall -y virtualenv && pip install virtualenv

export CUDA_VISIBLE_DEVICES=""

# Fetch external dependencies (including Eigen)
bazel fetch "//tensorflow/... -//tensorflow/contrib/nccl/... -//tensorflow/examples/android/..."
bash third_party/patch_eigen_for_cuda9.sh

bazel test  -c opt --verbose_failures \
            --test_tag_filters=local,-benchmark-test \
            --test_env=CUDA_VISIBLE_DEVICES \
            -- \
            //tensorflow/... \
            -//tensorflow/compiler/... \
            -//tensorflow/python/kernel_tests:benchmark_test \
  | tee testresult.tmp && grep "test\.log" testresult.tmp \
  | /opt/tensorflow/qa/show_testlogs
