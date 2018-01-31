#!/bin/bash

set -e
python /opt/tensorflow/nvidia-examples/OpenSeq2Seq/test/create_reversed_examples.py


# 1GPU Test
python /opt/tensorflow/nvidia-examples/OpenSeq2Seq/run.py --config_file=/opt/tensorflow/nvidia-examples/OpenSeq2Seq/example_configs/toy_data_config.json --mode=train --logdir=ModelAndLogFolder_1GPU

python /opt/tensorflow/nvidia-examples/OpenSeq2Seq/run.py --config_file=/opt/tensorflow/nvidia-examples/OpenSeq2Seq/example_configs/toy_data_config.json --mode=infer --logdir=ModelAndLogFolder_1GPU --inference_out=pred_1GPU.txt

Result_1GPU="$(/opt/tensorflow/nvidia-examples/OpenSeq2Seq/multi-bleu.perl test/toy_data/test/target.txt < pred_1GPU.txt)"
echo $Result_1GPU

BLEU_1GPU=$(echo $Result_1GPU | awk '{print $3}' | sed 's/,$//')

python <<EOF
min_score = 95.0
if $BLEU_1GPU > min_score:
  print("PASS")
  exit(0)
else:
  print("FAIL, BLEU must be greater than ", min_score)
  exit(1)
EOF


# 2GPU Test
python /opt/tensorflow/nvidia-examples/OpenSeq2Seq/run.py --config_file=/opt/tensorflow/nvidia-examples/OpenSeq2Seq/example_configs/toy_data_config_2GPUs.json --mode=train --logdir=ModelAndLogFolder_2GPU

python /opt/tensorflow/nvidia-examples/OpenSeq2Seq/run.py --config_file=/opt/tensorflow/nvidia-examples/OpenSeq2Seq/example_configs/toy_data_config_2GPUs.json --mode=infer --logdir=ModelAndLogFolder_2GPU --inference_out=pred_2GPU.txt

Result_2GPU="$(/opt/tensorflow/nvidia-examples/OpenSeq2Seq/multi-bleu.perl test/toy_data/test/target.txt < pred_2GPU.txt)"
echo $Result_2GPU

BLEU_2GPU=$(echo $Result_2GPU | awk '{print $3}' | sed 's/,$//')

python <<EOF
min_score = 95.0
if $BLEU_2GPU > min_score:
  print("PASS")
  exit(0)
else:
  print("FAIL, BLEU must be greater than ", min_score)
  exit(1)
EOF


echo "ALL TESTS PASS"
