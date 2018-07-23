CURRENT_DIR=`pwd`

#Script Dir is tensorflow/jetson/qa/L0_tftrt/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR

cd ../../../nvidia-examples/tftrt/
python example.py
cd ../../tensorflow/contrib/tensorrt/test/
python test_tftrt.py
python tf_trt_integration_test.py
cd $CURRENT_DIR
