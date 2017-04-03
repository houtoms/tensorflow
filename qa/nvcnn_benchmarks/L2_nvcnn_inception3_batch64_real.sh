#!/bin/bash

export CNN_MODEL=inception3
export CNN_BATCH_SIZE=64
export CNN_DATA_DIR="/data/imagenet/train-val-tfrecord"

./test.sh
