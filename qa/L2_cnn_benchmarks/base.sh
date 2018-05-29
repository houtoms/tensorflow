#!/bin/bash

DATA_DIR="${DATA_DIR:-/data/imagenet/train-val-tfrecord-480}"
USING_DATA_DIR=0
WARMUP=50
RUN_TO=300 #Best if multiple of WARMUP

function usage {
    cat<<EOF

Usage $0 <CNN_SCRIPT> [BATCH1[,FILTER1[,FILTER2...]] BATCH2...]

Runs cross product of benchmarks for each listed batch size.

Default Dimensions:
      GPUS     1, MAX_GPUS
      DATA      real, fake
 Precision      fp16, fp32

The following filters can be chained (comma separated) onto each
batch to limit the scan space.

    REAL_ONLY
    FAKE_ONLY
    FP16_ONLY
    FP32_ONLY
    <K>GPUS_ONLY
    <N>MiB_MIN
    <N>MiB_MAX

EOF
}


function run_config {
    local CNN_SCRIPT="$1"
    local BATCH="$2"
    local MATH="$3"
    local DATA="$4"
    local GPUS="$5"
    if [[ $? -ne 0 ]]; then
        echo Failed to create temp file for stdout.
        exit 1
    fi


    for D in $DATA; do
        local DATA_FLAG=""
        if [[ "$D" == "real" ]]; then
            DATA_FLAG="--data_dir=$DATA_DIR"
        fi

        for M in $MATH; do
            for G in $GPUS; do
                if [[ "$G" -gt "$MAX_GPUS" ]]; then continue; fi

                BASE_NAME="$(echo "${CNN_SCRIPT##*/}" | sed -e 's/ //g' -e 's/-layers=//g' -e 's/.py\(-\|$\)/\1/g')"
                if [[ -n "$LOG_DIR" ]]; then
                    local TMPFILE="$LOG_DIR/${BASE_NAME}_${D}_${M}_${BATCH}x${G}.log"
                else
                    local TMPFILE="$(mktemp tmp.XXXXXX)"
                fi

                echo mpiexec --bind-to socket --allow-run-as-root -np $G python -u \
                    $CNN_SCRIPT \
                    $DATA_FLAG \
                    --precision=$M \
                    --num_iter=$RUN_TO \
                    --iter_unit=batch \
                    --display_every=$WARMUP \
                    --batch=$BATCH > "$TMPFILE"
                echo "==================================================" >> "$TMPFILE"
            
                SECONDS=0
                mpiexec --allow-run-as-root --bind-to socket -np $G python -u \
                    $CNN_SCRIPT \
                    $DATA_FLAG \
                    --precision=$M \
                    --num_iter=$RUN_TO \
                    --iter_unit=batch \
                    --display_every=$WARMUP \
                    --batch=$BATCH >> "$TMPFILE" 2>&1
                WALLTIME=$SECONDS

                if [[ $? -ne 0 ]]; then
                    if [[ -z "$LOG_DIR" ]]; then
                        cat "$TMPFILE"
                        echo TRAINING SCRIPT FAILED
                        rm -f "$TMPFILE"
                        exit 1
                    fi
                    local PERF="FAIL"
                else
                    local PERF=$(cat "$TMPFILE" | grep "^ *[0-9]\+ " | awk "
                        {
                            if (\$1 == $((WARMUP*2))) {start=1};
                            if (start == 1) {sum += \$3; count+=1}
                        }
                        END {if (count == $((RUN_TO/WARMUP-1))) {print sum/count}}")

                    if [[ -z "$PERF" ]]; then
                        if [[ -z "$LOG_DIR" ]]; then
                            cat "$TMPFILE"
                            echo UNEXPECTED END OF LOG
                            rm -f "$TMPFILE"
                            exit 1
                        fi
                        PERF="FAIL"
                    fi
                fi
                [[ -z "$LOG_DIR" ]] && rm -f "$TMPFILE"

                if [[ "$PERF" == "FAIL" ]]; then
                    printf "%-30s %4s %4s %5d %4d %9s %8d\n" "$BASE_NAME" $D $M $BATCH $G $PERF $WALLTIME
                else
                    printf "%-30s %4s %4s %5d %4d %9.3f %8d\n" "$BASE_NAME" $D $M $BATCH $G $PERF $WALLTIME
                fi
            done
        done
    done
    
}

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

AVAIL_MiB=$(nvidia-smi -q -d MEMORY \
            | grep -A3 "FB Memory Usage" \
            | grep Total \
            | awk 'BEGIN {F=1; M=-1}
                   {
                       if ($4 != "MiB") {M=-1}
                       if (F==1 || $3<M) {M=$3; F=0}
                   }
                   END {print M}')
if [[ "$AVAIL_MiB" == "-1" ]]; then
    echo "Failed to detect GPU memory sizes"
    exit 1
fi

MAX_GPUS=$(nvidia-smi -L | wc -l)

CNN_SCRIPT=""
declare -a BATCHES
declare -a MATHS
declare -a DATAS
declare -a DEVS

while [[ $# -gt 0 ]]; do

    if [[ "$1" == "--help" || "$1" == "-h" ]]; then
        usage
        exit 0
    elif [[ -z "$CNN_SCRIPT" ]]; then
        CNN_SCRIPT="$1"
    else
        CONFIG="$1"
        BATCH=$(echo $CONFIG | cut -d',' -f1)
        FILTERS=${CONFIG#$BATCH}
        MATH="fp16 fp32"
        DATA="real fake"
        GPUS="1 $MAX_GPUS"
        MIN_MiB=-1
        MAX_MiB=-1

        if [[ -z "$BATCH" || ! "$BATCH" =~ ^[0-9]*$ ]]; then
            usage
            echo Invalid batch size $BATCH in config $CONFIG
            exit 1
        fi

        for F in $(echo $FILTERS | sed 's/,/ /g'); do
            case "$F" in
                REAL_ONLY) DATA="real" ;;
                FAKE_ONLY) DATA="fake" ;;
                FP16_ONLY) MATH="fp16" ;;
                FP32_ONLY) MATH="fp32" ;;
                *GPUS_ONLY)
                    GPUS=${F%GPUS_ONLY}
                    if [[ "$GPUS" == "MAX" ]]; then
                        GPUS=$MAX_GPUS
                    elif [[ ! "$GPUS" =~ ^[0-9]*$ ]]; then
                        usage
                        echo Invalid GPU filter $F
                        exit 1
                    fi
                    ;;
                *MiB_MIN)
                    MIN_MiB=${F%MiB_MIN}
                    if [[ ! "$MIN_MiB" =~ ^[0-9]*$ ]]; then
                        usage
                        echo Invalid MiB_MIN filter $F
                        exit 1
                    fi
                    ;;
                *MiB_MAX)
                    MAX_MiB=${F%MiB_MAX}
                    if [[ ! "$MAX_MiB" =~ ^[0-9]*$ ]]; then
                        usage
                        echo Invalid MiB_MAX filter $F
                        exit 1
                    fi
                    ;;
                *) usage; echo Invalid filter $F in config $CONFIG; exit 1 ;;
            esac
        done

        if [[ ! "$MIN_MiB" =~ ^[0-9]*|-1$ ]]; then
            usage
            echo "Invalid MiB ($MIN_MiB) in config $CONFIG"
            exit 1
        fi
        if [[ ! "$MAX_MiB" =~ ^[0-9]*|-1$ ]]; then
            usage
            echo "Invalid MiB ($MAX_MiB) in config $CONFIG"
            exit 1
        fi


        if [[ ( "$MIN_MiB" -eq -1 || "$AVAIL_MiB" -ge "$MIN_MiB" ) && 
              ( "$MAX_MiB" -eq -1 || "$AVAIL_MiB" -lt "$MAX_MiB" ) ]]; then
            BATCHES+=("$BATCH")
            MATHS+=("$MATH")
            DATAS+=("$DATA")
            DEVS+=("$GPUS")
        fi
    fi
    shift
done

if [[ -z "$CNN_SCRIPT" ]]; then
    usage
    echo "No <CNN_SCRIPT> provided."
    exit 1
fi

if [[ "$SKIP_HEADER" -eq 0 ]]; then
    echo '--------------------------------------------------------------------------------'
    echo TensorFlow Container ${NVIDIA_TENSORFLOW_VERSION:-N/A}
    echo Container Build ID ${NVIDIA_BUILD_ID:-N/A}
    if [[ "$USING_DATA_DIR" -eq 1 ]]; then
        echo Data from $DATA_DIR
    fi
    sed -n 's/^model name\s*:\s*\([^\s].*\)$/\1/p' /proc/cpuinfo | head -n 1
    echo Uptime: $(uptime)
    nvidia-smi
    echo '--------------------------------------------------------------------------------'
    printf "%-30s %4s %4s %5s %4s %9s %8s\n" NETWORK DATA MATH BATCH GPUs IMG/SEC WALLSECS
fi


for i in "${!BATCHES[@]}"; do
  run_config "$CNN_SCRIPT" "${BATCHES[$i]}" "${MATHS[$i]}" "${DATAS[$i]}" "${DEVS[$i]}"
done

if [[ "$SKIP_FOOTER" -eq 0 ]]; then
    echo All tests complete.
fi
exit 0
