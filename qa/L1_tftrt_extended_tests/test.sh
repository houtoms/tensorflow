#!/bin/bash

# Setup for TF research/slim
# https://github.com/tensorflow/models/tree/master/research/slim
ln -sf ../third_party/tensorflow_models models
cd models/research/slim
python setup.py install
cd ../../..

# Setup for TF models/official
# https://github.com/tensorflow/models/tree/master/official
CURRENT_PATH=$(pwd)
export PYTHONPATH="$PYTHONPATH:$CURRENT_PATH/models"
pip install requests

set -e


models=( mobilenet_v1 mobilenet_v2 nasnet_large nasnet_mobile resnet_v1_50 resnet_v2_50 vgg_16 vgg_19 inception_v3 inception_v4 )

###################### TF_TRT INFERENCE TESTS #####################

for i in "${models[@]}"
do
  python inference.py --model $i --use_trt > output_$i
  #python check_accuracy.py --model $i --input output_$i
done

#################### TENSORFLOW INFERENCE TESTS ########################

for i in "${models[@]}"
do
  python inference.py --model $i > output_$i
  #python check_accuracy.py --model $i --input output_$i
done
