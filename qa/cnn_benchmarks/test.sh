#!/bin/bash

CNN_NUM_BATCHES=300
CNN_DATA_FORMAT=NCHW
CNN_RESIZE_METHOD=bilinear
CNN_DISPLAY_EVERY=10
CNN_NUM_PREPROCESS_THREADS=5
CNN_SHARED_CONFIG=" --nodistortions --weak_scaling"
CNN_NUM_GPUS_LIST=${CNN_NUM_GPUS_LIST:-"1 2 4"}
CNN_PARAMETER_SERVER=${CNN_PARAMETER_SERVER:-"cpu"}

cd ..

for n in ${CNN_NUM_GPUS_LIST//;/ }; do
    python models/benchmark_tf_cnn.py \
        --num_gpus=$n \
        --model=$CNN_MODEL \
        --batch_size=$CNN_BATCH_SIZE \
        --num_batches=$CNN_NUM_BATCHES \
        --parameter_server=$CNN_PARAMETER_SERVER \
        --data_format=$CNN_DATA_FORMAT \
        --resize_method=$CNN_RESIZE_METHOD \
        --display_every=$CNN_DISPLAY_EVERY \
        --num_preprocess_threads=$CNN_NUM_PREPROCESS_THREADS \
        ${CNN_DATA_DIR:+"--data_dir=$CNN_DATA_DIR"} \
        $CNN_SHARED_CONFIG \
        $CNN_CONFIG \
        || exit; \
done
