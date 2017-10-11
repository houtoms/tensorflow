#!/bin/bash

set -e

nvidia-smi
cd /opt/tensorflow
./nvbuild.sh --configonly --python2

tensorflow/tools/ci_build/install/install_bootstrap_deb_packages.sh
add-apt-repository -y ppa:openjdk-r/ppa && \
  add-apt-repository -y ppa:george-edison55/cmake-3.x
tensorflow/tools/ci_build/install/install_deb_packages.sh

pip  install wheel six==1.10.0 protobuf==3.0.0 numpy==1.11.0 scipy==0.16.1 scikit-learn==0.17.1 pandas==0.18.1 psutil py-cpuinfo pylint pep8 portpicker mock

pip3 install wheel six==1.10.0 protobuf==3.0.0 numpy==1.11.0 scipy==0.16.1 scikit-learn==0.17.1 pandas==0.18.1 psutil py-cpuinfo pylint pep8 portpicker

tensorflow/tools/ci_build/install/install_pip_packages.sh
tensorflow/tools/ci_build/install/install_proto3.sh
tensorflow/tools/ci_build/install/install_auditwheel.sh

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Note: //tensorflow/python/debug:debugger_cli_common_test fails when run as root due to a file permissions issue.
# Note: //tensorflow/contrib/tensor_forest:scatter_add_ndim_op_test fails for an unknown reason with "Create kernel failed: Invalid argument: AttrValue must not have reference type value of float_ref".
# Note: //tensorflow/contrib/distributions:mvn_full_covariance_test fails due to assert_equal being used to check symmetry of the result of a matmul.
NUM_GPUS=`nvidia-smi -L | wc -l` && \
  bazel test  --config=cuda -c opt --verbose_failures --local_test_jobs=$NUM_GPUS \
              --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute \
              --test_tag_filters=-no_gpu,-benchmark-test \
              --build_tests_only \
              -- \
              //tensorflow/... \
              //tensorflow/contrib/cudnn_rnn:cudnn_rnn_ops_test \
              //tensorflow/contrib/cudnn_rnn:cudnn_rnn_ops_test_cc \
              -//tensorflow/python/kernel_tests:benchmark_test \
              -//tensorflow/compiler/... \
              -//tensorflow/go/... \
              -//tensorflow/python/debug:debugger_cli_common_test \
              -//tensorflow/contrib/tensor_forest:scatter_add_ndim_op_test \
              -//tensorflow/contrib/distributions:mvn_full_covariance_test \
              -//tensorflow/python:localhost_cluster_performance_test \
              -//tensorflow/core/debug:grpc_session_debug_test \
  | tee testresult.tmp

# Note: These two tests were observed to fail intermittently with error
#       "address already in use" when run as part of the above command
#       on a DGX-1.
bazel test    --config=cuda -c opt --verbose_failures --local_test_jobs=1 \
              --test_tag_filters=-no_gpu,-benchmark-test \
              --build_tests_only \
              -- \
              //tensorflow/python:localhost_cluster_performance_test \
              //tensorflow/core/debug:grpc_session_debug_test \
  | tee -a testresult.tmp

grep "test\.log" testresult.tmp | /opt/tensorflow/qa/show_testlogs
