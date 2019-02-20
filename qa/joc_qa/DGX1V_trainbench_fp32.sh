#!/bin/bash

python ./qa/testscript.py --mode training --precision fp32 --bench-warmup 200 --bench-iterations 500 --ngpus 1 4 8 --bs 32 64 128 --baseline qa/benchmark_baselines/RN50_tensorflow_train_fp32.json --data_dir $1 --results_dir $2