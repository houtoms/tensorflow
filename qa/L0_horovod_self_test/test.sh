#!/bin/bash

set -e

cd /opt/tensorflow/third_party/horovod/test/
python -m unittest test_tensorflow.py
