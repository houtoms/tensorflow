
# Convolutional neural network training script

This script implements a number of popular CNN models and demonstrates
efficient training on multi-GPU systems. It can be used for benchmarking,
training and evaluation of models.

## Script usage

Benchmarking example (assuming TFRecord dataset in /data/imagenet_tfrecord):

    $ python nvcnn.py --model=resnet50 \
                      --data_dir=/data/imagenet_tfrecord \
                      --batch_size=64 \
                      --num_gpus=8

Training example:

    $ python nvcnn.py --model=resnet50 \
                      --data_dir=/data/imagenet_tfrecord \
                      --batch_size=64 \
                      --num_gpus=8 \
                      --num_epochs=120 \
                      --display_every=50 \
                      --log_dir=/home/train/resnet50-1

Add `--eval` to the arguments to evaluate a trained model on the validation
dataset. Run with `--help` to see additional arguments.

TensorBoard can be used to monitor training:

    $ tensorboard --logdir=/home/train

## Script details

### Supported models
| Key | Name | Paper |
| alexnet                | AlexNet 'One Weird Trick'  | https://arxiv.org/abs/1404.5997  |
| googlenet              | GoogLeNet                  | https://arxiv.org/abs/1409.4842  |
| vgg11,13,16,19         | Visual Geometry Group ABDE | https://arxiv.org/abs/1409.1556  |
| resnet18,34,50,101,152 | Residual Networks v1       | https://arxiv.org/abs/1512.03385 |
| inception3             | Inception v3               | https://arxiv.org/abs/1512.00567 |
| inception4             | Inception v4               | https://arxiv.org/abs/1602.07261 |
| inception-resnet2      | Inception-ResNet v2        | https://arxiv.org/abs/1602.07261 |

### Image transformations
The image input pipeline performs the following operations:
 * Random crop and resize
 * Random horizontal flip
 * Random color distortions

### Optimizations
The key optimizations used in this script are:
 * Use `tf.parallel_stack` to construct batches of images.
     * This encodes the parallelism in the graph itself instead of relying on
       Python threads, which are not as efficient as TF's backend thread-pool.
 * Use `StagingArea` to stage input data in host and device memory, and
   explicitly pre-fill them before training begins.
     * This enables overlap of IO and PCIe operations with computation.
 * Use NCHW data format throughout the model.
     * This allows efficient CUDNN convolutions to be used.
 * Use the fused batch normalization op.
     * This is faster than the non-fused version.
 * Apply XLA `jit_scope` to groups of simple bandwidth-bound ops.
     * This allows the ops to be fused together, reducing the number of kernel
       launches and round-trips through memory.
