#!/bin/bash

set -e
cd ../../nvidia-examples/OpenSeq2Seq

PYMAJ=$(python -c 'import sys; print(sys.version_info[0])')
if [[ $PYMAJ -eq 2 ]]; then
  echo "Open Seq2Seq requires Python 3. Skipping test."
  exit 0
fi

python -m unittest discover -s open_seq2seq -p '*_test.py'

