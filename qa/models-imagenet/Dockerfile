FROM nvdl.githost.io:4678/dgx/tensorflow:17.02-stage
MAINTAINER NVIDIA CORPORATION <cudatools@nvidia.com>

WORKDIR /opt

COPY models-imagenet/test_result test_result

CMD python -m tensorflow.models.image.imagenet.classify_image | grep score | ./test_result
