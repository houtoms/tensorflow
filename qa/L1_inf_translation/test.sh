#!/bin/bash

set -e
set -v
set -o pipefail

DATA_DIR="/data/wmt16_en_de_OpenSeq2Seq/"
# Temporary files
TEMP_OUTPUT_FILE="raw.txt"
TEMP_DECODED_OUTPUT_FILE="decoded.txt"

# Model configs
# "trt_" will be prepended to these names for config files with TF-TRT enabled.
model_configs=(
  "convs2s_config.py"
  "transformer_config.py"
)

PYMAJ=$(python -c 'import sys; print(sys.version_info[0])')
if [[ $PYMAJ -eq 2 ]]; then
  echo "Open Seq2Seq requires Python 3. Skipping test."
  exit 0
fi

TEST_DIR="$(pwd)"
pushd /opt/tensorflow/nvidia-examples/OpenSeq2Seq

run_and_check() {
  MODEL_CONFIG="$2${model_configs[$1]}"
  RESULT_FILE="$2${model_configs[$1]}.result"
  LOG_FILE="$2${model_configs[$1]}.log"
  # Run inference
  common_args="--config_file="$TEST_DIR/$MODEL_CONFIG" \
                --mode=infer \
                --infer_output_file=$TEMP_OUTPUT_FILE"
  python -u run.py $common_args 2>&1 | tee $LOG_FILE
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
                                         --model $MODEL_CONFIG
  echo "DONE testing $MODEL_CONFIG"
}

for ((i=0; i<${#model_configs[@]}; ++i))
do
  run_and_check $i
  # Disable TF-TRT temporarily.
  # run_and_check $i "trt_"
done

popd
