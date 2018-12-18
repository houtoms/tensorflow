#!/bin/bash

MNIST=../../tensorflow/examples/tutorials/mnist

#without xla
python $MNIST/mnist_softmax_xla.py --xla="" 2> /dev/null | tail -n 1 | ./test_result
#with xla
python $MNIST/mnist_softmax_xla.py  2> /dev/null | tail -n 1 | ./test_result
