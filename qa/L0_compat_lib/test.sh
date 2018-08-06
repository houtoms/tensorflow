#!/bin/bash
set -e

g++ -o ctx -I/usr/local/cuda/include ctx.cpp -L/usr/local/cuda/lib64 -lcuda
./ctx
