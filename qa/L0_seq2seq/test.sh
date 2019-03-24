#!/bin/bash

set -e
cd ../../nvidia-examples/OpenSeq2Seq

#TODO(nluehr) Remove CUDA_VISIBLE_DEVICE restriction once
# [B] 2523310 is fixed.
export CUDA_VISIBLE_DEVICES=0

PYMAJ=$(python -c 'import sys; print(sys.version_info[0])')
if [[ $PYMAJ -eq 2 ]]; then
  echo "Open Seq2Seq requires Python 3. Skipping test."
  exit 0
fi

python -m unittest discover -s open_seq2seq -p '*_test.py'

