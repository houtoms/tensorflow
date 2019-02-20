set -o nounset
set -o errexit
set -o pipefail

cd /opt/tensorflow/qa/joc_qa

# Setup dataset directory
if [ ! -d "imagenet-tfrecord" ]; then
   ln -sf /data/imagenet/train-val-tfrecord/ imagenet-tfrecord
fi

export PYTHONPATH=/opt/tensorflow/nvidia-examples/resnet50v1.5/

mkdir /tmp/results

python ./testscript.py --mode training --precision fp16 --bench-warmup 200 --bench-iterations 500 --ngpus 1 4 8 --bs 64 128 256 --baseline qa/benchmark_baselines/RN50_tensorflow_train_fp16.json --data_dir imagenet-tfrecord --results_dir /tmp/results

