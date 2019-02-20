set -o nounset
set -o errexit
set -o pipefail

cd /opt/tensorflow/qa/joc_qa

# Setup dataset directory
if [ ! -d "imagenet-tfrecord" ]; then
   ln -sf /data/imagenet/train-val-tfrecord/ imagenet-tfrecord
fi

export PYTHONPATH=/opt/tensorflow/nvidia-examples/resnet50v1.5/

mkdir -p /tmp/results

python ./qa/test_accuracy.py --precision fp16 --iterations 90 --ngpus 8 --bs 128 --top1-baseline 76 --top5-baseline 92 --data_dir imagenet-tfrecord --results_dir /tmp/results
