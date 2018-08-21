#!/bin/bash

exec ./base.sh "/workspace/nvidia-examples/cnn/vgg.py --layers=11" \
    32 \
    64,12000MiB_MIN,16000MiB_MAX \
    128,16000MiB_MIN
