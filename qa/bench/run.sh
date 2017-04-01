#!/bin/bash

function bench {
    NET=$1
    BATCH=$2
    NGPU=$3
    ITER=$4
    PS=$5
    CONFIG=$6
    NET_NAME=$7
    NUM_PROC_PER_GPU=$(expr 40 / $NGPU)
    echo Running $NET, batchsize $BATCH, $NGPU GPUs, $ITER iterations, parameter server on $PS
    #python -u models/benchmark_tf_cnn.py --model $NET_NAME --batch_size $BATCH --num_preprocess_threads $NUM_PROC_PER_GPU \
    #                            --num_gpus $NGPU --num_batches $ITER --parameter_server $PS $CONFIG
    python -u ../../nvidia-examples/cnn/nvcnn.py \
      --model=$NET_NAME \
      --batch_size=$BATCH \
      --num_gpus=$NGPU \
      --num_batches=$ITER \
      $CONFIG
}

# Sets model-specific args
function set_model_args {
    MODEL=$1

    case "$MODEL" in

    googlenet)
        BATCHES_PER_GPU=(32 64 )
        PS=cpu
        NET_NAME=googlenet
        ;;
    vgg_11)
        BATCHES_PER_GPU=(32 64 )
        PS=gpu
        NET_NAME=vgg11
        ;;
    vgg_16)
        BATCHES_PER_GPU=(32 64 )
        PS=cpu
        NET_NAME=vgg16
        ;;
    vgg_19)
        BATCHES_PER_GPU=(32 64 )
        PS=cpu
        NET_NAME=vgg19
        ;;
    overfeat)
        BATCHES_PER_GPU=(32 64 )
        PS=gpu
        NET_NAME=overfeat
        ;;
    alexnet_owt)
        BATCHES_PER_GPU=(128)
        PS=gpu
        NET_NAME=alexnet
       ;;
    inception_v3)
        BATCHES_PER_GPU=(32 64 )
        PS=cpu
        NET_NAME=inception3
       ;;
    inception_v4)
        BATCHES_PER_GPU=(32 64 )
        PS=cpu
        NET_NAME=inception4
       ;;
    resnet_50)
        BATCHES_PER_GPU=(32 64 )
        PS=cpu
        NET_NAME=resnet50
       ;;
    resnet_101)
        BATCHES_PER_GPU=(32 64 )
        PS=cpu
        NET_NAME=resnet101
       ;;
    resnet_152)
        BATCHES_PER_GPU=(32 64 )
        PS=cpu
        NET_NAME=resnet152
       ;;
    inception-resnet_v2)
        BATCHES_PER_GPU=(32 64 )
        PS=cpu
        NET_NAME=inception-resnet2
       ;;
    esac
}

TIMESTAMP=$(date +%m%d%H%M)
export THIS_DIR=$(cd $(dirname $0); pwd)
export TESTS_DIR=${TESTS_DIR:-"${THIS_DIR}"}
export RESULTS_DIR=${RESULTS_DIR:-"${TESTS_DIR}/results"}
export LOG_DIR=${LOG_DIR:-"${RESULTS_DIR}/logs"}
export CSV_DIR=${CSV_DIR:-"${RESULTS_DIR}/csv"}
mkdir -p $LOG_DIR
echo "Clear out old logs..."
rm -rf ${LOG_DIR}/*.log
echo "Keeping a global log in full.log (each run will also have a separate log..."
exec > >(tee "$LOG_DIR/full.log") 2>&1
SYSTEM_GPUS="$(nvidia-smi -L | wc -l)"
export MAXGPUS=${MAXGPUS:-$SYSTEM_GPUS}
echo "Max GPUs = $MAXGPUS"
echo "Running nvidia-smi -L" | tee    $LOG_DIR/nvidia_smi.log
nvidia-smi -L                | tee -a $LOG_DIR/nvidia_smi.log
echo "Running nvidia-smi"    | tee -a $LOG_DIR/nvidia_smi.log
nvidia-smi                   | tee -a $LOG_DIR/nvidia_smi.log
echo "Running nvidia-smi -a" | tee -a $LOG_DIR/nvidia_smi.log
nvidia-smi -a                | tee -a $LOG_DIR/nvidia_smi.log

# Synthetic data is not currently supported
#DATA="--data_dir=/imagenet"
DATA="--data_dir=/media/bbarsdel/bendisk/data/bbarsdel/imagenet_tensorflow_short480_quality85"

CONFIG="
    --display_every=200
    $DATA
"
#Dryrun to cache imagenet
MODEL=alexnet_owt
set_model_args $MODEL
# Number of iterations * batchsize should be equal to dataset size
BATCH=128
NGPU=1
ITER=10000
PS=gpu
echo Dryrun $MODEL, batchsize $BATCH, $NGPU GPUs, $ITER iterations
bench "$MODEL" "$BATCH" "$NGPU" "$ITER" "$PS" "$CONFIG" "$NET_NAME" 2>&1 | tee ${LOG_DIR}/dryrun_${MODEL}_b${BATCH}_${NGPU}gpu.log
echo 'Done with dryrun.'

CONFIG="
    --display_every=20
    $DATA
"
MODELS=(googlenet vgg_11 vgg_16 vgg_19 alexnet_owt inception_v3 inception_v4 resnet_50 resnet_101 resnet_152 inception-resnet_v2)

echo 'Running Benchmark...'
for MODEL in ${MODELS[@]}; do
    GPUS=(1 2 4 8)
    ITER=300
    set_model_args $MODEL
    for BATCH_PER_GPU in ${BATCHES_PER_GPU[@]}; do
        for NGPU in ${GPUS[@]}; do
            set_model_args $MODEL
            if [[ $NGPU = 1 ]]; then
                PS=gpu
            fi
            BATCH=$(expr $BATCH_PER_GPU \* $NGPU)
            bench "$MODEL" "$BATCH_PER_GPU" "$NGPU" "$ITER" "$PS" "$CONFIG" "$NET_NAME" 2>&1 | tee ${LOG_DIR}/output_${MODEL}_b${BATCH}_${NGPU}gpu.log
        done
    done
done
echo 'Done with Benchmark.'

echo 'Running parser...'
python ${THIS_DIR}/parsers/parser.py ${LOG_DIR}/output*.log

echo 'CSV file:'
cat bench.csv

mkdir -p ${CSV_DIR}
mv bench.csv ${CSV_DIR}/bench.${TIMESTAMP}.csv
mkdir -p ${LOG_DIR}/logs.${TIMESTAMP}
mv ${LOG_DIR}/*.log ${LOG_DIR}/logs.${TIMESTAMP}/
