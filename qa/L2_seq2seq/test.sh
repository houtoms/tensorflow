#!/bin/bash

set -e
SEQ2SEQ_NUM_GPUS_LIST=${SEQ2SEQ_NUM_GPUS_LIST:-"1 2 4"}
SEQ2SEQ_BATCH_LIST=${SEQ2SEQ_BATCH_LIST:-"1 64 128"}
TEST_LEN=20
SUMMARY_FREQ=100
TARGET_DIR=${1:-/opt/tensorflow/nvidia-examples/OpenSeq2Seq}
OUT_DIR=${2:-/opt/tensorflow/qa/L2_seq2seq}
DATA_DIR=${3:-/data/wmt16_en_de}
MAX_GPUS=`nvidia-smi -L | wc -l`

for n in ${SEQ2SEQ_NUM_GPUS_LIST//;/ }; do
    if [[ $n -gt $MAX_GPUS ]]; then
        continue
    fi
    for b in ${SEQ2SEQ_BATCH_LIST//;/ }; do
       #skip benchmark when batch size is smaller then number of gpus
       if [[ $n -gt $b ]]; then
           continue
       fi
       source $TARGET_DIR/try_gnmt_en2de.sh $OUT_DIR $n $b $SUMMARY_FREQ $DATA_DIR $TEST_LEN
    done
done
