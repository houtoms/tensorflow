#!/bin/bash

python ./qa/testscript.py --mode training --precision fp16 --bench-warmup 200 --bench-iterations 500 --ngpus 1 4 8 --bs 64 128 256 --baseline qa/benchmark_baselines/RN50_tensorflow_train_fp16.json  --data_dir $1 --results_dir $2