#!/bin/bash

set -e
cd ../../nvidia-examples/OpenSeq2Seq


PYMAJ=$(python -c 'import sys; print(sys.version_info[0])')
if [[ $PYMAJ -eq 2 ]]; then
  echo "Open Seq2Seq requires Python 3. Skipping test."
  exit 0
fi

#TODO(nluehr) Return to all tests once [B] 2523310 is fixed.
TEST_TO_PATCH=open_seq2seq/models/speech2text_ds2_test.py
PATCHED=$(grep "unittest.skip" $TEST_TO_PATCH | wc -l)
if [[ $PATCHED -eq 0 ]]; then
    sed -e '0,/^import tensorflow/s/^import tensorflow.*/import unittest\n&/' \
        -e 's/^\( *\)\(def test_regularizer(\)/\1@unittest.skip\n\1\2/' \
        -e 's/^\( *\)\(def test_convergence(\)/\1@unittest.skip\n\1\2/' \
        -i $TEST_TO_PATCH
fi

python -m unittest discover -s open_seq2seq -p '*_test.py'
