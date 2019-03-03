#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/outputs"
mkdir -p $OUTPUT_DIR
TEST_LIST="$OUTPUT_DIR/tests.list"
TARGETS="tests(//tensorflow/python:amp_optimizer_test)"

current_arch=`uname -m`
if [[ $current_arch == "aarch64" ]]; then
  pip install portpicker
  export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
  ARM_EXCEPT="union attr(tags, no_arm, $TARGETS)"
else
  ARM_EXCEPT=""
fi

cd $SCRIPT_DIR/../..
bazel query "attr(size, small, $TARGETS) union attr(size, medium, $TARGETS)" > "$TEST_LIST"

GPUS=$(nvidia-smi -L | wc -l)
echo Running tests on $GPUS GPUs
SECONDS=0

set +x
set +e
set -o pipefail

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

for i in $(seq 0 $((GPUS-1))); do
  {
    FAIL_COUNT=0
    PASS_COUNT=0
    export CUDA_VISIBLE_DEVICES=$i
    while read -r TEST; do
      TEST_NAME=${TEST##*:}
      TEST_FILE="tensorflow/python/grappler/$TEST_NAME.py"
      LOG_FILE="$OUTPUT_DIR/$TEST_NAME.out"
      if [[ -f "$TEST_FILE" ]]; then
        python "$TEST_FILE" &> "$LOG_FILE"
        if [[ $? -eq 0 ]]; then
          echo "PASS   $TEST_NAME"
          PASS_COUNT=$((PASS_COUNT+1))
        else
          cat "$LOG_FILE"
          echo "FAIL  $TEST_NAME"
          FAIL_COUNT=$((FAIL_COUNT+1))
        fi
      fi
    done < <(sed -n "$((i+1))~$GPUS p" "$TEST_LIST") && echo $PASS_COUNT $FAIL_COUNT>"$OUTPUT_DIR/stats.$i"
  } &
done

wait
trap - SIGINT SIGTERM EXIT

TOT_FAILS=0
TOT_PASSES=0
for f in $(ls $OUTPUT_DIR/stats.*); do
  STATS=$(cat $f)
  PASSES=${STATS% *}
  FAILS=${STATS#* }
  TOT_PASSES=$((TOT_PASSES+PASSES))
  TOT_FAILS=$((TOT_FAILS+FAILS))
done

TOT_TESTS=$((TOT_PASSES+TOT_FAILS))
echo RAN $TOT_TESTS TESTS IN $SECONDS SECONDS.
echo $TOT_FAILS TESTS FAILED.

[[ $TOT_FAILS -eq 0 ]] && exit 0 || exit 1

