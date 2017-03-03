#!/bin/bash

cd /opt/tensorflow
curl -O https://bootstrap.pypa.io/get-pip.py && \
  python get-pip.py && \
  python3 get-pip.py && \
  pip2 install --upgrade --force-reinstall pip && \
  rm get-pip.py

pip  install wheel six==1.10.0 protobuf==3.0.0 numpy==1.11.0 scipy==0.16.1 scikit-learn==0.17.1 pandas==0.18.1 psutil py-cpuinfo pylint pep8 portpicker mock

pip3 install wheel six==1.10.0 protobuf==3.0.0 numpy==1.11.0 scipy==0.16.1 scikit-learn==0.17.1 pandas==0.18.1 psutil py-cpuinfo pylint pep8 portpicker

tensorflow/tools/ci_build/install/install_bootstrap_deb_packages.sh
add-apt-repository -y ppa:openjdk-r/ppa && \
  add-apt-repository -y ppa:george-edison55/cmake-3.x
tensorflow/tools/ci_build/install/install_deb_packages.sh
tensorflow/tools/ci_build/install/install_pip_packages.sh
tensorflow/tools/ci_build/install/install_proto3.sh
tensorflow/tools/ci_build/install/install_auditwheel.sh

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

NUM_GPUS=`nvidia-smi -L | wc -l` && \
  bazel build --config=cuda -c opt --test_tag_filters=local,-benchmark-test \
              //tensorflow/... \
              -//tensorflow/compiler/... && \
  bazel test  --config=cuda -c opt --verbose_failures --local_test_jobs=$NUM_GPUS \
              --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute \
              --test_tag_filters=local,-benchmark-test \
              -- \
              //tensorflow/... \
              -//tensorflow/python/kernel_tests:atrous_conv2d_test \
              -//tensorflow/python/kernel_tests:benchmark_test \
              -//tensorflow/compiler/... \
  | tee testresult.tmp && grep test.log testresult.tmp \
  | /opt/tensorflow/qa/show_testlogs
