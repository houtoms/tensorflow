#!/bin/bash
set -e

cd /opt/tensorflow/nvidia-examples/NCF/
bash ./qa/test_dgx1v16g_fp32.sh
