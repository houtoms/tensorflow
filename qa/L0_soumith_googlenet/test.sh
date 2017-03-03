
cd ../third_party/convnet-benchmarks/tensorflow
python benchmark_googlenet.py | tee /dev/tty | grep -q "Forward-backward across 100 steps"
