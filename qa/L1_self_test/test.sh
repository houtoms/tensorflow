
cd /opt/tensorflow
tensorflow/tools/ci_build/install/install_bootstrap_deb_packages.sh
add-apt-repository -y ppa:openjdk-r/ppa && \
  add-apt-repository -y ppa:george-edison55/cmake-3.x
tensorflow/tools/ci_build/install/install_deb_packages.sh
tensorflow/tools/ci_build/install/install_pip_packages.sh
tensorflow/tools/ci_build/install/install_proto3.sh
tensorflow/tools/ci_build/install/install_auditwheel.sh

NUM_GPUS=`nvidia-smi -L | wc -l` && \
  bazel build --config=cuda -c opt --test_tag_filters=local,-benchmark-test //tensorflow/... && \
  bazel test  --config=cuda -c opt --verbose_failures --local_test_jobs=$NUM_GPUS --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute --test_tag_filters=local,-benchmark-test //tensorflow/... \
  | tee testresult.tmp && grep test.log testresult.tmp | /opt/tensorflow/qa/L1_self_test/show_testlogs
