#!/bin/bash

set -e
cd ../../nvidia-examples/OpenSeq2Seq

python -m unittest discover -s open_seq2seq -p '*_test.py'

