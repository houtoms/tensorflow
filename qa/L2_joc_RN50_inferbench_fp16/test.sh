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

python ./testscript.py --mode inference --precision fp16 --bench-warmup 100 --bench-iterations 200 --ngpus 1 --bs 1 2 4 8 16 32 64 128 256 --baseline benchmark_baselines/RN50_tensorflow_infer_fp16.json  -data_dir imagenet-tfrecord --results_dir /tmp/results
