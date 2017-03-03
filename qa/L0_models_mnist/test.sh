#!/bin/bash

python /opt/tensorflow/tensorflow/examples/tutorials/mnist/mnist_softmax.py 2> /dev/null | tail -n 1 | ./test_result
