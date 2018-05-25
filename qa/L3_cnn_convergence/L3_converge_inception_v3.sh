#!/bin/bash

GPUS=$(nvidia-smi -L | wc -l)
if [[ "$GPUS" -lt 8 ]]; then
    echo "Error convergence test requires 8 GPUs. Found only $GPUS"
    exit 1
fi

LOG=$(mktemp XXXXXX.log)
OUT=${LOG%.log}.dir

function CLEAN_AND_EXIT {
    rm -rf $LOG $OUT
    exit $1
}

SECONDS=0
mpiexec --allow-run-as-root --bind-to socket -np 8 python -u \
    /opt/tensorflow/nvidia-examples/cnn/inception_v3.py \
    --data_dir=/data/imagenet/train-val-tfrecord-480 \
    --log_dir=$OUT --display_every=1000 2>&1 | tee $LOG
RET=${PIPESTATUS[0]}
echo "Training ran in $SECONDS seconds"
if [[ $RET -ne 0 ]]; then
    echo "Error in training script."
    CLEAN_AND_EXIT 2
fi

MIN_TOP1=77.0
MIN_TOP5=93.0

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
