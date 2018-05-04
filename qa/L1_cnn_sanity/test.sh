#!/bin/bash

echo '--------------------------------------------------------------------------------'
echo TensorFlow Container $NVIDIA_TENSORFLOW_VERSION
echo Container Build ID $NVIDIA_BUILD_ID
nvidia-smi
echo Uptime: $(uptime)
echo '--------------------------------------------------------------------------------'

GPUS=$(nvidia-smi -L | wc -l)
BATCH_SIZE=32
PRECISION="fp32"
DATA="--data_dir=/data/imagenet/train-val-tfrecord-480"

NETWORKS=("alexnet.py" \
          "googlenet.py" \
          "inception_v3.py" \
          "overfeat.py" \
          "resnet.py --layers=50" \
          "resnet.py --layers=101" \
          "resnet.py --layers=152" \
          "vgg.py --layers=11" \
          "vgg.py --layers=16" \
          "vgg.py --layers=19" )

get_PERF() {
    SCRIPT="$1"
    mpiexec --allow-run-as-root -np $GPUS python -u \
        /tf/nvidia-examples/cnn/$SCRIPT \
        --num_iter=101 \
        --iter_unit=batch \
        --display_every=50 \
        $DATA \
        --batch=$BATCH_SIZE \
        --precision=$PRECISION &> log.tmp
    
    if [[ $? -ne 0 ]]; then
        cat log.tmp
        echo TRAINING SCRIPT FAILED
        exit 1
    fi
    
    PERF=$(grep "^ *100 " log.tmp | awk '{print $3}')

    if [[ -z "$PERF" ]]; then
        cat log.tmp
        echo UNEXPECTED END OF LOG
        exit 1
    fi
}

for net in "${NETWORKS[@]}"; do
    get_PERF "$net"
    name=$(echo $net | sed 's/.py --layers=//')
    name=${name%.py}
    echo "${name} $PERF img/sec with $GPUs GPUs * $BATCH_SIZE img/gpu"
done
echo All tests pass.
exit 0

