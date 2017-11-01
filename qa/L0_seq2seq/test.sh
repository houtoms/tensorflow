#!/bin/bash

set -e
cd /opt/tensorflow/nvidia-examples/OpenSeq2Seq
./create_toy_data.sh
python -m unittest test.data_layer_tests
python -m unittest test.model_tests

echo "ALL TESTS PASS"
