#!/bin/bash

python -m tensorflow.models.image.imagenet.classify_image | grep score | ./test_result
