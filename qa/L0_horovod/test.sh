#!/bin/bash

set -e
cd /opt/tensorflow/nvidia-examples/cnn

MAX_GPUS=$(nvidia-smi -L | wc -l)
GPUS=1

while [[ $GPUS -le $MAX_GPUS ]]; do
    echo -n "$GPUS: "
    R=$(mpiexec --allow-run-as-root -np $GPUS \
            python nvcnn_hvd.py --model resnet50 \
                                --num_batches 50 \
                                --batch_size 64 \
                                --display_every 10 \
            2>&1 | \
        grep "^Images" | awk '{print $2}')
    [[ "$R" =~ ^[0-9]*[.][0-9]$ ]] || (echo FAILED && false)
    echo $R "img/sec"
    GPUS=$((GPUS*2))
done
