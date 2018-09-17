#!/bin/bash
set -e
set -o pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
TEST_LIST="$THIS_DIR/../../tensorflow/compiler/tests/tests.list"
GPUS=$(nvidia-smi -L | wc -l)
NUM_TESTS=$(wc -l "$TEST_LIST" | cut -d' ' -f1)
rm -rf "$THIS_DIR/outputs"
mkdir "$THIS_DIR/outputs"
TESTS_PER_GPU=$(( ($NUM_TESTS + $GPUS - 1) / $GPUS ))
cd "$THIS_DIR/../../tensorflow/compiler"
export TEST_SRCDIR=$THIS_DIR
rm -f $THIS_DIR/org_tensorflow
ln -s $THIS_DIR/../.. $THIS_DIR/org_tensorflow

echo Running $NUM_TESTS xla tests on $GPUS GPUs

for i in $(seq 0 $((GPUS-1))); do
    {
        FAILS=0
        export CUDA_VISIBLE_DEVICES=$i
        while read -r INDEX SHARDS SCRIPT ARGS; do
            SCRIPT="${SCRIPT%%.py*}"
            NAME="${SCRIPT##*/}"
            SCRIPT="${SCRIPT%_[gc]pu}.py"

            if [[ "$INDEX" != "NONE" ]]; then
                export TEST_SHARD_INDEX=$INDEX
                NAME=${NAME}_$((INDEX+1))
            fi
            if [[ "$SHARDS" != "NONE" ]]; then
                export TEST_TOTAL_SHARDS=$SHARDS
                NAME=${NAME}_OF_$SHARDS
            fi

            sed 's/^ *from tensorflow\.compiler\./from /' $SCRIPT | \
                python - $ARGS &> "$THIS_DIR/outputs/$NAME"
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
