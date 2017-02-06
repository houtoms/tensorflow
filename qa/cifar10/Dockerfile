FROM nvdl.githost.io:4678/dgx/tensorflow:17.02-stage
MAINTAINER NVIDIA CORPORATION <cudatools@nvidia.com>

WORKDIR /usr/local/lib/python2.7/dist-packages/tensorflow/models/image/cifar10

COPY cifar10/test_result test_result

CMD python cifar10_multi_gpu_train.py --num_gpus=4 --max_steps=15000 && \
    python cifar10_eval.py --run_once | grep precision | ./test_result
