#!/bin/bash
set +x
set +e
set -o pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
NATIVE_ARCH=`uname -m`
TEST_LIST="$THIS_DIR/../../tensorflow/python/kernel_tests/tests.list"
if [ ${NATIVE_ARCH} == 'aarch64' ]; then
    export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
    pip install portpicker
    JPVER=$(${THIS_DIR}/../../jetson/get_jpver.sh)
    cp "$THIS_DIR/../../wheelhouse/$JPVER/kernel_tests/tests.list" "$THIS_DIR/../../tensorflow/python/kernel_tests/tests.list"
    GPUS=1
else
    GPUS=$(nvidia-smi -L | wc -l)
fi
NUM_TESTS=$(wc -l "$TEST_LIST" | cut -d' ' -f1)
rm -rf "$THIS_DIR/outputs"
mkdir "$THIS_DIR/outputs"
TESTS_PER_GPU=$(( ($NUM_TESTS + $GPUS - 1) / $GPUS ))
cd "$THIS_DIR/../.."
export TEST_SRCDIR=$THIS_DIR
rm -f $THIS_DIR/org_tensorflow
ln -s $PWD $THIS_DIR/org_tensorflow

echo Running $NUM_TESTS kernel tests on $GPUS GPUs

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

            python $SCRIPT $ARGS &> "$THIS_DIR/outputs/$NAME"
            if [[ $? -eq 0 ]]; then
                echo PASS -- $NAME
            else
                FAILS=$((FAILS+1))
                tail -n 30 "$THIS_DIR/outputs/$NAME"
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
