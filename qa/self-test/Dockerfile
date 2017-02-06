FROM nvdl.githost.io:4678/dgx/tensorflow:17.02-stage
MAINTAINER NVIDIA CORPORATION <cudatools@nvidia.com>

WORKDIR /opt/tensorflow
RUN tensorflow/tools/ci_build/install/install_bootstrap_deb_packages.sh
RUN add-apt-repository -y ppa:openjdk-r/ppa &&     add-apt-repository -y ppa:mc3man/trusty-media &&     add-apt-repository -y ppa:george-edison55/cmake-3.x
RUN tensorflow/tools/ci_build/install/install_deb_packages.sh

COPY install_pip_packages.sh install_pip_packages.sh
RUN ./install_pip_packages.sh
RUN tensorflow/tools/ci_build/install/install_proto3.sh
RUN tensorflow/tools/ci_build/install/install_auditwheel.sh

COPY show_testlogs show_testlogs

CMD NUM_GPUS=`nvidia-smi -L | wc -l` && bazel build --config=cuda -c opt --test_tag_filters=local,-benchmark-test //tensorflow/... && bazel test --config=cuda -c opt --verbose_failures --local_test_jobs=$NUM_GPUS --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute --test_tag_filters=local,-benchmark-test //tensorflow/... | tee testresult.tmp && grep test.log testresult.tmp | ./show_testlogs
