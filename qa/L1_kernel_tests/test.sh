#!/bin/bash
set -e
set -o pipefail

TEST_LIST="/opt/tensorflow/tensorflow/python/kernel_tests/tests.list"
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
GPUS=$(nvidia-smi -L | wc -l)
NUM_TESTS=$(wc -l "$TEST_LIST" | cut -d' ' -f1)
rm -rf "$THIS_DIR/outputs"
mkdir "$THIS_DIR/outputs"
TESTS_PER_GPU=$(( ($NUM_TESTS + $GPUS - 1) / $GPUS ))
cd /opt/tensorflow
export TEST_SRCDIR=$THIS_DIR
rm -f $THIS_DIR/org_tensorflow
ln -s /opt/tensorflow $THIS_DIR/org_tensorflow

echo Running $NUM_TESTS kernel tests on $GPUS GPUs

for i in $(seq 0 $((GPUS-1))); do
    {
        FAILS=0
        export CUDA_VISIBLE_DEVICES=$i
        while read -r INDEX SHARDS LINE; do
            SCRIPT="${LINE%%.py*}"
            NAME="${SCRIPT##*/}"

            if [[ "$INDEX" != "NONE" ]]; then
                export TEST_SHARD_INDEX=$INDEX
                NAME=${NAME}_$((INDEX+1))
            fi
            if [[ "$SHARDS" != "NONE" ]]; then
                export TEST_TOTAL_SHARDS=$SHARDS
                NAME=${NAME}_OF_$SHARDS
            fi

            python $LINE &> "$THIS_DIR/outputs/$NAME"
            if [[ $? -eq 0 ]]; then
                echo PASS -- $NAME
            else
                FAILS=$((FAILS+1))
                echo FAIL -- $NAME
            fi
            unset TEST_TOTAL_SHARDS TEST_SHARD_INDEX
        done < <(sed -n "$((i+1))~$GPUS p" "$TEST_LIST") && echo $FAILS>"$THIS_DIR/outputs/fails.$i"
    } &
done

wait

TOT_FAILS=0
for f in "$THIS_DIR"/outputs/fails.*; do
    T=$(cat $f)
    TOT_FAILS=$((TOT_FAILS+$T))
done

echo "Ran $NUM_TESTS tests. $TOT_FAILS failed."
[[ $TOT_FAILS -eq 0 ]] && exit 0 || exit 1
