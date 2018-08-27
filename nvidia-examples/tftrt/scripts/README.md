# TensorFlow-TensorRT Examples

This script will run inference using a few popular image classification models on the ImageNet validation set.

You can turn on TensorFlow-TensorRT integration with the flag `--use_trt`. This will apply TensorRT inference optimization to speed up execution for portions of the model's graph where supported, and will fall back to native TensorFlow for layers and operations which are not supported. See https://devblogs.nvidia.com/tensorrt-integration-speeds-tensorflow-inference/ for more information.

## Models

This test supports the following models for image classification:
* MobileNet v1
* MobileNet v2
* NASNet - Large
* NASNet - Mobile
* ResNet50 v1
* ResNet50 v2
* VGG16
* VGG19
* Inception v3
* Inception v4

## Setup

* If running inside DGX/tensorflow docker container, simply run `source setup.sh` with the default options.

* Otherwise:

  * Clone [tensorflow/models](https://github.com/tensorflow/models): `git clone https://github.com/tensorflow/models.git`

  * Run `source setup.sh` and provide the path to where you cloned the tensorflow/models directory when prompted.

Note: the PYTHONPATH environment variable will be not be saved between different shells. You can either rerun the setup script each time you work in a new shell, or add
`export PYTHONPATH="$PYTHONPATH:$MODELS"` to your .bashrc file (replacing $MODELS with the path to your tensorflow/models repository).

### Data

By default, the script will look for ImageNet validation files in TFRecord format stored in the directory `/data/imagenet/train-val-tfrecord`. You can change the directory by using `--data_dir /path/to/your/TFRecords`. The validation TFRecords should be named according to the pattern: `validation-*-of-00128`.

You can download and process Imagenet using [this script provided by TF Slim](https://github.com/tensorflow/models/blob/master/research/slim/datasets/download_imagenet.sh). Please note that this script downloads both the training and validation sets, and this example only requires the validation set.

## Usage

`python inference.py --model vgg16 [--use_trt]`

### All Options

```
usage: inference.py [-h]
                    [--model {mobilenet_v1,mobilenet_v2,nasnet_mobile,nasnet_large,resnet_v1_50,resnet_v2_50,vgg_16,vgg_19,inception_v3,inception_v4}]
                    [--data_dir DATA_DIR] [--calib_data_dir CALIB_DATA_DIR]
                    [--use_trt] [--precision {fp32,fp16,int8}]
                    [--batch_size BATCH_SIZE]
                    [--num_iterations NUM_ITERATIONS]
                    [--display_every DISPLAY_EVERY] [--use_synthetic]
                    [--num_warmup_iterations NUM_WARMUP_ITERATIONS]
                    [--num_calib_inputs NUM_CALIB_INPUTS]

Evaluate model

optional arguments:
  -h, --help            show this help message and exit
  --model {mobilenet_v1,mobilenet_v2,nasnet_mobile,nasnet_large,resnet_v1_50,resnet_v2_50,vgg_16,vgg_19,inception_v3,inception_v4}
                        Which model to use.
  --data_dir DATA_DIR   Directory containing validation set TFRecord files.
  --calib_data_dir CALIB_DATA_DIR
                        Directory containing TFRecord files for calibrating
                        int8.
  --use_trt             If set, the graph will be converted to a TensorRT
                        graph.
  --precision {fp32,fp16,int8}
                        Precision mode to use. FP16 and INT8 only work in
                        conjunction with --use_trt
  --batch_size BATCH_SIZE
                        Number of images per batch.
  --num_iterations NUM_ITERATIONS
                        How many iterations(batches) to evaluate. If not
                        supplied, the whole set will be evaluated.
  --display_every DISPLAY_EVERY
                        Number of iterations executed between two consecutive
                        display of metrics
  --use_synthetic       If set, one batch of random data is generated and used
                        at every iteration.
  --num_warmup_iterations NUM_WARMUP_ITERATIONS
                        Number of initial iterations skipped from timing
  --num_calib_inputs NUM_CALIB_INPUTS
                        Number of inputs (e.g. images) used for calibration
                        (last batch is skipped in case it is not full)

```
