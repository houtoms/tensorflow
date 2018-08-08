# TensorFlow-TensorRT Examples

This script tests a few popular models on the ImageNet validation set, to ensure that inference accuracy remains the same after converting the model using TensorFlow-TensorRT.

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

test.sh performs the required setup.

## Usage

`./test.sh`