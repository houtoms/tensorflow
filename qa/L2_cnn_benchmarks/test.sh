#!/bin/bash

# Sweep through benchmark tests for all networks.

set -e
nvidia-smi
export SKIP_NVIDIA_SMI=1
for NET in L2_bench_*.sh; do
    ./$NET
done
