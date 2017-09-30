#!/bin/bash

export CNN_MODEL=resnet50
export CNN_BATCH_SIZE=64
export CNN_DATA_DIR="/data/imagenet/train-val-tfrecord"
export CNN_NUM_BATCHES=50
export CNN_NUM_GPUS_LIST="2"

./test.sh
