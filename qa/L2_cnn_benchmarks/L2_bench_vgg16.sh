#!/bin/bash

exec ./base.sh "/workspace/nvidia-examples/cnn/vgg.py --layers=16" \
    32 \
    64,16000_MiB_MIN,32000MiB_MAX \
    128,32000MiB_MIN
