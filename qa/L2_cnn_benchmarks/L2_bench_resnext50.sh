#!/bin/bash

exec ./base.sh "/workspace/nvidia-examples/cnn/resnext.py --layers=50" 32,MAXGPUS_ONLY,FP32_ONLY,FAKE_ONLY
