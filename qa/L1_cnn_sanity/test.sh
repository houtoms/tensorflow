#!/bin/bash

# Script has explicit error logging
set +x
set +e

echo '--------------------------------------------------------------------------------'
echo TensorFlow Container $NVIDIA_TENSORFLOW_VERSION
echo Container Build ID $NVIDIA_BUILD_ID
echo Uptime: $(uptime)
echo '--------------------------------------------------------------------------------'

GPUS=$(nvidia-smi -L 2>/dev/null| wc -l || echo 1)
BATCH_SIZE=32
PRECISION="fp32"
DATA="--data_dir=/data/imagenet/train-val-tfrecord-480"

NETWORKS=("alexnet.py" \
          "googlenet.py" \
          "inception_v3.py" \
          "inception_v4.py" \
          "inception_resnet_v2.py" \
          "overfeat.py" \
          "resnet.py --layers=50" \
          "resnet.py --layers=101" \
          "resnet.py --layers=152" \
          "vgg.py --layers=11" \
          "vgg.py --layers=16" \
          "vgg.py --layers=19" \
          )

get_PERF() {
    SCRIPT="$1"
    MAX_RUNS=2
    for RUN in $(seq 1 $MAX_RUNS); do
        local TMP_DIR="$(mktemp -d tmp.XXXXXX)"
        mpiexec --bind-to none --allow-run-as-root -np $GPUS python -u \
            ../../nvidia-examples/cnn/$SCRIPT \
            --num_iter=101 \
            --iter_unit=batch \
            --display_every=50 \
            $DATA \
            --log_dir="$TMP_DIR" \
            --batch=$BATCH_SIZE \
            --precision=$PRECISION &> log.tmp
        local RET=$?
        rm -rf "$TMP_DIR"

        if [[ $RET -eq 0 ]]; then
            PERF=$(grep "^ *100 " log.tmp | awk '{print $3}')
            [[ -z "$PERF" ]] && RET=1
        fi

        NANLOSS=$(grep NanLossDuringTrainingError log.tmp | wc -l)
        if [[ $RET -eq 0 || $NANLOSS -eq 0 ]]; then
            break
        fi
    done

    if [[ $RET -ne 0 ]]; then
        cat log.tmp
        echo TRAINING SCRIPT FAILED FOR $SCRIPT ON RUN $RUN
        exit 1
    fi
    rm log.tmp
}

for net in "${NETWORKS[@]}"; do
    get_PERF "$net"
    name=$(echo $net | sed 's/.py --layers=//')
    name=${name%.py}
    echo "${name} $PERF img/sec with $GPUS GPUs * $BATCH_SIZE img/gpu"
done
echo All tests pass.
exit 0

