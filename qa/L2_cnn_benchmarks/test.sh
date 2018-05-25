#!/bin/bash

# Sweep through benchmark tests for all networks.

set -e

for NET in L2_bench_*.sh; do
    ./$NET
    export SKIP_HEADER=1
done

echo All tests complete.
exit 0
