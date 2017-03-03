#!/bin/bash

pip install future
cd ../third_party/convnet-benchmarks/tensorflow
python benchmark_overfeat.py
