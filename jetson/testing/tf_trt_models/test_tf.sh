#!/bin/bash



#python tf_inference_test.py --model resnet_v2_50 --num_classes 1001

echo "Test inception_v1"
python tf_inference_test.py --model inception_v1 #>/dev/null
if diff cur_resinception_v1 res1inception_v1 2>/dev/null; then
  echo "FAILED"
else
  echo "PASSED"
fi


echo "Test inception_v2"
python tf_inference_test.py --model inception_v2 #>/dev/null
if diff cur_resinception_v2 res1inception_v2 2>/dev/null; then
  echo "FAILED"
else
  echo "PASSED"
fi



echo "Test inception_v3"
python tf_inference_test.py --model inception_v3
if diff cur_resinception_v3 res1inception_v3 2>/dev/null; then
  echo "FAILED"
else
  echo "PASSED"
fi


echo "Test inception_v4"
python tf_inference_test.py --model inception_v4
if diff cur_resinception_v4 res1inception_v4 2>/dev/null; then
  echo "FAILED"
else
  echo "PASSED"
fi




echo "Test resnet_v1_50"
python tf_inference_test.py --model resnet_v1_50 --num_classes 1000
if diff cur_resresnet_v1_50 res1resnet_v1_50 2>/dev/null; then
  echo "FAILED"
else
  echo "PASSED"
fi


echo "Test resnet_v1_101"
python tf_inference_test.py --model resnet_v1_101 --num_classes 1000
if diff cur_resresnet_v1_101 res1resnet_v1_101 2>/dev/null; then
  echo "FAILED"
else
  echo "PASSED"
fi


echo "Test resnet_v1_152"
python tf_inference_test.py --model resnet_v1_152 --num_classes 1000
if diff cur_resresnet_v1_152 res1resnet_v1_152 2>/dev/null; then
  echo "FAILED"
else
  echo "PASSED"
fi

#python tf_inference_test.py --model resnet_v2_101 --num_classes 1000
#python tf_inference_test.py --model resnet_v2_152 --num_classes 1000

#python tf_inference_test.py --model mobilenet_v1_0.25_128 --num_classes 1001
