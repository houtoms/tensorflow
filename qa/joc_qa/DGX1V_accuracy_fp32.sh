#!/bin/bash

mkdir -p /tmp/results
python ./qa/test_accuracy.py --precision fp32 --iterations 90 --ngpus 8 --bs 128 --top1-baseline 76 --top5-baseline 92 --data_dir $1 --results_dir $2
