#!/bin/bash

exec ./base.sh "/workspace/nvidia-examples/cnn/resnet.py --layers=50" \
    32 \
    64,12000MiB_MAX,FP16_ONLY \
    128,12000MiB_MIN,16000MiB_MAX,FP16_ONLY \
    256,16000MiB_MIN,FP16_ONLY \
    64,16000MiB_MAX,FP32_ONLY \
    128,16000MiB_MIN,32000MiB_MAX,FP32_ONLY \
    256,32000MiB_MIN,FP32_ONLY
