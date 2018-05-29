#!/bin/bash

# Sweep through benchmark tests for all networks.

if [[ $# -gt 0 ]]; then
    export LOG_DIR="$1"
    mkdir -p "$LOG_DIR" &> /dev/null
    if [[ $? -ne 0 ]]; then
        echo "Aborting: Failed to access LOG_DIR $LOG_DIR."
        exit 1
    fi
    if [[ -n "$(ls -A "$LOG_DIR")" ]]; then
        echo "Aborting: LOG_DIR $LOG_DIR is not empty."
        exit 1
    fi
    RESULT_FILE="$LOG_DIR/test.out"
else
    export LOG_DIR=""
    RESULT_FILE=/dev/null
fi

set -e

export SKIP_FOOTER=1

for NET in L2_bench_*.sh; do
    ./$NET 2>&1 | tee -a "$RESULT_FILE"
    export SKIP_HEADER=1
done

echo All tests complete.
exit 0
