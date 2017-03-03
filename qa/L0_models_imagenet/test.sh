#!/bin/bash

IMAGENET=../third_party/tensorflow_models/tutorials/image/imagenet

python $IMAGENET/classify_image.py | grep score | ./test_result
