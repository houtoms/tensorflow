#!/bin/bash

set +x

PYMAJ=$(python -c 'import sys; print(sys.version_info.major)')
if [[ "$PYMAJ" -lt 3 ]]; then
  echo "OpenSeq2Seq requires Python 3."
  echo "Test skipped."
  exit 0
fi

cd /workspace/nvidia-examples/OpenSeq2Seq
python ctc_decoder_with_lm/ctc-test.py
