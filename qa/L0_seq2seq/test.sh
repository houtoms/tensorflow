#!/bin/bash

set -e
cd /opt/tensorflow/nvidia-examples/OpenSeq2Seq

python -m unittest discover -s open_seq2seq -p '*_test.py'

