#!/bin/bash

set -e

cd ../../third_party/horovod/test/
python -m unittest test_tensorflow.py
