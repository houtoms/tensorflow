#!/bin/bash

set -e
set -v
set -o pipefail

DATA_DIR="/data/wmt16_en_de_OpenSeq2Seq/"
MODEL_DIR="/data/tensorflow/machine_translation/models"
# Temporary files
TEMP_OUTPUT_FILE="raw.txt"
TEMP_DECODED_OUTPUT_FILE="decoded.txt"

# Model config
model_logdirs=(
  "Transformer-FP32-H-256"
)
model_configs=(
  "transformer-bp-fp32.py"
)

PYMAJ=$(python -c 'import sys; print(sys.version_info[0])')
if [[ $PYMAJ -eq 2 ]]; then
  echo "Open Seq2Seq requires Python 3. Skipping test."
  exit 0
fi

TEST_DIR="$(pwd)"
pushd /opt/tensorflow/nvidia-examples/OpenSeq2Seq

run_and_check() {
  MODEL_LOGDIR="${model_logdirs[$1]}"
  MODEL_CONFIG="${model_configs[$1]}"
  RESULT_FILE="result_${model_logdirs[$1]}"
  LOG_FILE="log_${model_logdirs[$1]}"
  # Run inference
  common_args="--config_file="$MODEL_DIR/$MODEL_LOGDIR/$MODEL_CONFIG" \
                --mode=infer \
                --infer_output_file=$TEMP_OUTPUT_FILE \
                --logdir=$MODEL_DIR/$MODEL_LOGDIR"
  python -u run.py $common_args $2 2>&1 | tee $LOG_FILE
  # Detokenize BPE segments into words
  python tokenizer_wrapper.py --mode=detokenize \
                              --model_prefix="$DATA_DIR/m_common" \
                              --decoded_output=$TEMP_DECODED_OUTPUT_FILE \
                              --text_input=$TEMP_OUTPUT_FILE
  # Get BLEU score
  cat $TEMP_DECODED_OUTPUT_FILE | sacrebleu -t wmt14 -l en-de > $RESULT_FILE
  # Parse outputs and check score
  python -u "$TEST_DIR/check_results.py" --input_result $RESULT_FILE \
                                         --input_log $LOG_FILE \
                                         --model $MODEL_LOGDIR
  echo "DONE testing $MODEL_LOGDIR"
}

for ((i=0; i<${#model_logdirs[@]}; ++i))
do
  run_and_check $i
  run_and_check $i "--use_trt"
done

popd
