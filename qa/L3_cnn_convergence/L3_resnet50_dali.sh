#!/bin/bash

TOT_GPUS=$(nvidia-smi -L | wc -l)
if [[ "$TOT_GPUS" -ge 8 && "$TOT_GPUS" -lt 16 ]]; then
    GPUS=8
    BATCH_SIZE=256
elif [[ "$TOT_GPUS" -ge 16 ]]; then
    GPUS=16
    BATCH_SIZE=128
else
    echo "Error convergence test requires at least 8 GPUs. Found $GPUS"
    exit 1
fi
echo "Using $GPUS of $TOT_GPUS GPUs"

LOG=$(mktemp XXXXXX.log)
OUT=${LOG%.log}.dir

function CLEAN_AND_EXIT {
    rm -rf $LOG $OUT
    exit $1
}

SECONDS=0
mpiexec --allow-run-as-root --bind-to none -np $GPUS python -u \
    /opt/tensorflow/nvidia-examples/cnn/resnet.py --layers=50 \
    --data_dir=/data/imagenet/train-val-tfrecord-480 \
    --use_dali=GPU --data_idx_dir=/data/imagenet/train-val-tfrecord-480.idx \
    --batch_size=$BATCH_SIZE --log_dir=$OUT --display_every=1000 \
    2>&1 | tee $LOG
RET=${PIPESTATUS[0]}
echo "Training ran in $SECONDS seconds"
if [[ $RET -ne 0 ]]; then
    echo "Error in training script."
    CLEAN_AND_EXIT 2
fi

MIN_TOP1=75.0
MIN_TOP5=92.0

TOP1=$(grep "^Top-1" $LOG | awk '{print $3}')
TOP5=$(grep "^Top-5" $LOG | awk '{print $3}')

if [[ -z "$TOP1" || -z "$TOP5" ]]; then
    echo "Incomplete output."
    CLEAN_AND_EXIT 3
fi

TOP1_RESULT=$(echo "$TOP1 $MIN_TOP1" | awk '{if ($1>$2) {print "OK"} else { print "FAIL" }}')
TOP5_RESULT=$(echo "$TOP5 $MIN_TOP5" | awk '{if ($1>$2) {print "OK"} else { print "FAIL" }}')

echo
printf "TOP-1 Accuracy: %.2f%% (expect at least %f%%) %s\n" $TOP1 $MIN_TOP1 $TOP1_RESULT
printf "TOP-5 Accuracy: %.2f%% (expect at least %f%%) %s\n" $TOP5 $MIN_TOP5 $TOP5_RESULT

if [[ "$TOP1_RESULT" == "OK" && "$TOP5_RESULT" == "OK" ]]; then
    CLEAN_AND_EXIT 0
fi

CLEAN_AND_EXIT 4
