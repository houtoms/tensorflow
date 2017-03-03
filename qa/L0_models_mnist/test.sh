#!/bin/bash

MNIST=../../tensorflow/examples/tutorials/mnist

python $MNIST/mnist_softmax.py 2> /dev/null | tail -n 1 | ./test_result
