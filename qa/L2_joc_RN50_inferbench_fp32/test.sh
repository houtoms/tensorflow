set -o nounset
set -o errexit
set -o pipefail

cd /opt/tensorflow/nvidia-examples/resnet50v1.5/

# Setup dataset directory
if [ ! -d "imagenet-tfrecord" ]; then
   ln -sf /data/imagenet/train-val-tfrecord/ imagenet-tfrecord
fi

export PYTHONPATH=/opt/tensorflow/nvidia-examples/resnet50v1.5/

bash ./qa/DGX1V_inferbench_fp32.sh imagenet-tfrecord

