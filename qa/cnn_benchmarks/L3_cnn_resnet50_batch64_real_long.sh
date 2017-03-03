#!/bin/bash

export CNN_MODEL=resnet50
export CNN_BATCH_SIZE=64
export CNN_NUM_GPUS_LIST="4"
export CNN_NUM_BATCHES=100091
export CNN_DATA_DIR="/data/imagenet/train-val-tfrecord-352x352"

./test.sh
