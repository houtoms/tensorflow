#!/bin/bash

echo '--------------------------------------------------------------------------------'
echo TensorFlow Container $NVIDIA_TENSORFLOW_VERSION
echo Container Build ID $NVIDIA_BUILD_ID
echo Uptime: $(uptime)
echo '--------------------------------------------------------------------------------'

NATIVE_ARCH=`uname -m`
if [ ${NATIVE_ARCH} == 'aarch64' ]; then
  GPUS=1
else
  GPUS=$(nvidia-smi -L | wc -l)
fi
BATCH_SIZE=128
DATA="--data_dir=/data/imagenet/train-val-tfrecord-480 \
     --data_idx_dir=/data/imagenet/train-val-tfrecord-480.idx"

get_PERF() {
    PRECISION=$1
    
    mpiexec --bind-to socket --allow-run-as-root -np $GPUS python -u \
        ../../nvidia-examples/cnn/resnet.py \
        --use_dali \
        --layers=50 \
        --num_iter=100 \
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

get_PERF fp16
echo ResNet50 FP16 $PERF img/sec on $GPUS GPUs
exit 0

