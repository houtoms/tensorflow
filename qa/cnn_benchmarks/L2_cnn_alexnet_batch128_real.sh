#!/bin/bash

export CNN_MODEL=alexnet
export CNN_BATCH_SIZE=128
export CNN_DATA_DIR="/data/imagenet/train-val-tfrecord-352x352"

./test.sh
