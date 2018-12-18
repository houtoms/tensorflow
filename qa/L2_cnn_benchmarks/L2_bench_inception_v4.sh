#!/bin/bash

# For some reason synthetic data give NaN losses in some fp16 cases.

exec ./base.sh "/workspace/nvidia-examples/cnn/inception_v4.py" 32,REAL_ONLY 64,32000MiB_MIN,REAL_ONLY
