CURRENT_DIR=`pwd`

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR

cd ../../nvidia-examples/tftrt/
python example.py
cd ../../tensorflow/contrib/tensorrt/test/
python test_tftrt.py
python tf_trt_integration_test.py
cd $CURRENT_DIR
