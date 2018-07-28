#!/bin/bash

set -e
set -o pipefail

cd ../../nvidia-examples/OpenSeq2Seq
./create_toy_data.sh
python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-RR.py --mode=train_eval | tee nmt-reversal.out

python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-RR.py --mode=infer --infer_output_file=output.txt

RESULT=$(./multi-bleu.perl toy_text_data/test/target.txt < output.txt)

BLEU=$(echo $RESULT | awk '{print $3}' | sed 's/,$//')
PASS=$(echo BLEU | awk '{printf("%d", $1>0.9)}')
echo BLEU SCORE = $BLEU
echo MIN VALUE = 0.9
if [[ "$PASS" -eq 1 ]]; then
    echo PASS
    exit 0
else
    echo FAIL
    exit 1
fi

