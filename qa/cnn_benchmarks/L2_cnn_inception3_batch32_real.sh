#!/bin/bash

export CNN_MODEL=inception3
export CNN_BATCH_SIZE=32
export CNN_DATA_DIR="/data/imagenet/train-val-tfrecord-352x352"

./test.sh
