#!/bin/bash

CNN_NUM_BATCHES=300
CNN_DISPLAY_EVERY=10
CNN_SHARED_CONFIG=" "
CNN_NUM_GPUS_LIST=${CNN_NUM_GPUS_LIST:-"1 2 4"}

cd ..

for cnn_fp16_flag in "" "--fp16"; do
for n in ${CNN_NUM_GPUS_LIST//;/ }; do
    python ../nvidia-examples/cnn/nvcnn.py \
        --num_gpus=$n \
        --model=$CNN_MODEL \
        --batch_size=$CNN_BATCH_SIZE \
        --num_batches=$CNN_NUM_BATCHES \
        --display_every=$CNN_DISPLAY_EVERY \
        $cnn_fp16_flag \
        ${CNN_DATA_DIR:+"--data_dir=$CNN_DATA_DIR"} \
        $CNN_SHARED_CONFIG \
        $CNN_CONFIG \
        || exit; \
done
done
