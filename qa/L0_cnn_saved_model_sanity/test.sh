#!/bin/bash

set +x 
set -e

echo '--------------------------------------------------------------------------------'
echo TensorFlow Container $NVIDIA_TENSORFLOW_VERSION
echo Container Build ID $NVIDIA_BUILD_ID
echo Uptime: $(uptime)
echo '--------------------------------------------------------------------------------'

GPUS=$(nvidia-smi -L 2>/dev/null| wc -l || echo 1)
BATCH_SIZE=32
PRECISION="fp32"
DATA="--data_dir=/data/imagenet/train-val-tfrecord-480"

NETWORK="trivial.py"
MOD_RESTORE="saved_model_restore.py"

get_PERF() {
    SCRIPT="$1"
    MOD_SCRIPT="$2"
    local TMP_LOG_DIR="$(mktemp -d tmp.XXXXXX)"
    local TMP_MOD_DIR="$(mktemp -d tmp.XXXXXX)"
    echo Creating/Exporting the saved model
    mpiexec --bind-to none --allow-run-as-root -np $GPUS python -u \
        ../../nvidia-examples/cnn/$SCRIPT \
        --num_iter=101 \
        --iter_unit=batch \
        --display_every=50 \
        $DATA \
        --log_dir="$TMP_LOG_DIR" \
        --export_dir="$TMP_MOD_DIR" \
        --batch=$BATCH_SIZE \
        --precision=$PRECISION &> log.tmp
    RET=$?
    rm -rf "$TMP_LOG_DIR"

    if [[ $RET -ne 0 ]]; then
        cat log.tmp
        echo SAVED MODEL EXPORT SCRIPT FAILED FOR $SCRIPT
        exit 1
    fi

    echo Restoring the saved model
    local TMP_MOD_FILE="$(ls $TMP_MOD_DIR)"
    python -u ../../nvidia-examples/cnn/$MOD_SCRIPT \
        --export_dir=$TMP_MOD_DIR/$TMP_MOD_FILE &> log2.tmp
    RET=$?
    if [[ $RET -ne 0 ]]; then
        cat log2.tmp
        echo SAVED MODEL RESTORE SCRIPT FAILED FOR $MOD_SCRIPT
        exit 1
    fi

    rm -rf "$TMP_MOD_DIR"
    rm log.tmp log2.tmp
}

get_PERF "$NETWORK" "$MOD_RESTORE"

echo All tests pass.
exit 0

