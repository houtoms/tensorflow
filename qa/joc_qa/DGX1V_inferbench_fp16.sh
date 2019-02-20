#!/bin/bash

mkdir -p /tmp/results
python ./qa/testscript.py --mode inference --precision fp16 --bench-warmup 100 --bench-iterations 200 --ngpus 1 --bs 1 2 4 8 16 32 64 128 256 --baseline qa/benchmark_baselines/RN50_tensorflow_infer_fp16.json  --data_dir $1 --results_dir $2
