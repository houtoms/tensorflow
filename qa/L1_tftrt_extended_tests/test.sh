#!/bin/bash

PYTHONPATH=$PYTHONPATH:$PWD/../third_party/tensorflow_models/research/slim:$PWD/../third_party/tensorflow_models/research/

cd tf_trt_models


set -e


python tf_trt_inference_test.py --model inception_v1 --use_trt 1
rm -r data

python tf_trt_inference_test.py --model inception_v2 --use_trt 1
rm -r data

python tf_trt_inference_test.py --model inception_v3 --use_trt 1
rm -r data

python tf_trt_inference_test.py --model inception_v4 --use_trt 1
rm -r data



python tf_trt_inference_test.py --model resnet_v2_50 --num_classes 1001 --use_trt 1 
rm -r data

python tf_trt_inference_test.py --model resnet_v2_101 --num_classes 1001 --use_trt 1 
rm -r data

python tf_trt_inference_test.py --model resnet_v2_152 --num_classes 1001 --use_trt 1 
rm -r data




python tf_trt_inference_test.py --model resnet_v1_50 --num_classes 1001 --use_trt 1 
rm -r data

python tf_trt_inference_test.py --model resnet_v1_101 --num_classes 1001 --use_trt 1 
rm -r data

python tf_trt_inference_test.py --model resnet_v1_152 --num_classes 1001 --use_trt 1 
rm -r data




python tf_inference_test.py --model mobilenet_v1_0.25_128 --num_classes 1001 --use_trt 1
rm -r data

python tf_trt_inference_test.py --model mobilenet_v1_0p5_160 --num_classes 1001 --use_trt 1
rm -r data



python tf_trt_inference_test.py --model ssd_mobilenet_v1_coco --num_classes 1001 --use_trt 1 --detection 1
rm -r data
