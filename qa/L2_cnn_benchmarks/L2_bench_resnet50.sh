#!/bin/bash

exec ./base.sh "/workspace/nvidia-examples/cnn/resnet.py --layers=50" \
    32 \
    64 \
    128:12000:fp16 \
    128:16000:fp32 \
    256:16000:fp16 \
    256:32000:fp32
