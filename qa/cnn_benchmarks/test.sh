#!/bin/bash

set -e

CNN_NUM_BATCHES=300
CNN_DISPLAY_EVERY=10
CNN_NUM_GPUS_LIST=${CNN_NUM_GPUS_LIST:-"1 2 4"}

cd ..
MAX_GPUS=`nvidia-smi -L | wc -l`
for n in ${CNN_NUM_GPUS_LIST//;/ }; do
    if [[ $n -gt $MAX_GPUS ]]; then
        continue
    fi
    python -u /opt/tensorflow/nvidia-examples/cnn/nvcnn.py \
        --num_gpus=$n \
        --model=$CNN_MODEL \
        --batch_size=$CNN_BATCH_SIZE \
        --num_batches=$CNN_NUM_BATCHES \
        --display_every=$CNN_DISPLAY_EVERY \
        ${CNN_DATA_DIR:+"--data_dir=$CNN_DATA_DIR"}
done
