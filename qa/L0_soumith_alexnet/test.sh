
cd ../third_party/convnet-benchmarks/tensorflow
python benchmark_alexnet.py | tee /dev/tty | grep -q "Forward-backward across 100 steps"
