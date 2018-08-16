# TensorFlow-TensorRT Examples

This script will run inference on few popular models on the ImageNet validation set.

## Models

This test supports the following models for image classification:
* MobileNet v1
* MobileNet v2
* NASNet - Large
* NASNet - Mobile
* ResNet50 v1
* ResNet50 v2
* VGG16
* VGG19
* Inception v3
* Inception v4

## Setup

requirement.sh performs the required setup.
PYTHONPATH=$PYTHONPATH:$PWD/models will add to PYTHONPATH directory of official models

## Usage

`python inference.py --model vgg16 [--use_trt]`
