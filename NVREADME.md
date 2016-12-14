TensorFlow
============

## Introduction

**TensorFlow** is an open source software library for numerical computation using
data flow graphs.  Nodes in the graph represent mathematical operations, while
the graph edges represent the multidimensional data arrays (tensors) that flow
between them.  This flexible architecture lets you deploy computation to one
or more CPUs or GPUs in a desktop, server, or mobile device without rewriting
code.  TensorFlow also includes TensorBoard, a data visualization toolkit.

TensorFlow was originally developed by researchers and engineers
working on the Google Brain team within Google's Machine Intelligence research
organization for the purposes of conducting machine learning and deep neural
networks research.  The system is general enough to be applicable in a wide
variety of other domains, as well.

## Contents

This container has the TensorFlow Python package installed and ready to use.
/opt/tensorflow contains the complete source of this version of version of
TensorFlow.
 
## Getting Started

The basic command for running containers is to use the ```nvidia-docker run```
command, specifying the URL for the container, which includes the registry
address, repository name, and a version tag, similar to the following:
```$ nvidia-docker run nvcr.io/nvidia/tensorflow:16.12```
There are additional flags and settings that should be used with this command,
as described in the next sections.
 
### The Remove flag

By default, Docker containers remain on the system after being run.  Repeated
pull/run operations use up more and more space on the local disk, even after
exiting the container.  Therefore, it is important to clean up the Docker
containers after exiting.

To automatically remove a container when exiting, use the ```--rm``` flag:
```$ nvidia-docker run --rm nvcr.io/nvidia/tensorflow:16.12```

### Batch versus Interactive mode

By default, containers run in batch mode; that is, the container is run once
and then exited without any user interaction. Containers can also be run in
interactive mode.

To run in interactive mode, add the ```-ti``` flag to the run command:
```$ nvidia-docker run --rm -ti nvcr.io/nvidia/tensorflow:16.12```

To run in batch mode, leave out the ```-ti``` flag, and instead append the
command to be run in the container to the nvidia-docker run command line:
```$ nvidia-docker run --rm nvcr.io/nvidia/tensorflow:16.12 python myscript.py```

In both cases, it will often be desirable to pull in data and model
descriptions from locations outside the container (e.g., "myscript.py" in the
example above).  To accomplish this, the easiest method is to mount one or more
host directories as [Docker data volumes](https://docs.docker.com/engine/tutorials/dockervolumes/#/mount-a-host-directory-as-a-data-volume)
using the ```-v``` flag of ```nvidia-docker run```.

## Invoking TensorFlow

TensorFlow is run simply by importing it as a Python module:

```
$ python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> sess.run(hello)
Hello, TensorFlow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> sess.run(a+b)
42
```

## Suggested Reading

Please refer to the TensorFlow website at https://www.tensorflow.org/get_started
for a tutorial and basic usage examples.

This container image includes several built-in TensorFlow examples as well, which
can be run using commands like the following:
```
python -m tensorflow.models.image.mnist.convolutional
```
```
python -m tensorflow.models.image.cifar10.cifar10_multi_gpu_train
```

More information is available at the following sites:

* [TensorFlow website](http://tensorflow.org)
* [TensorFlow whitepaper](http://download.tensorflow.org/paper/whitepaper2015.pdf)
* [TensorFlow Model Zoo](https://github.com/tensorflow/models)
* [TensorFlow MOOC on Udacity](https://www.udacity.com/course/deep-learning--ud730)
* [Community-authored resources](https://www.tensorflow.org/versions/master/resources#community)

## Customizing the container

If you would like to modify the source code of the version of TensorFlow in this
container and run your customized version, or if you would like to install
additional packages into the container, you can easily use ```docker build``` to
add your customizations on top of this container.

First, create a file called "Dockerfile" describing the modifications you wish
to make -- see https://docs.docker.com/engine/reference/builder/ for a syntax
reference; some examples are below.

Then run a command like the following from the directory containing this new Dockerfile:
```
docker build -t my-custom-tensorflow:version .
```

This will allow you to ```nvidia-docker run ... my-custom-tensorflow:version ...``` in the
same way as you would otherwise have run the stock NVIDIA TensorFlow container.  Further,
it will allow you to "replay" your modifications on top of later NVIDIA TensorFlow containers
simply by updating the NVIDIA version tag in the "FROM" line of your Dockerfile and
rerunning ```docker build```.

### Adding packages
To install additional packages, create a Dockerfile similar to the following:
```
FROM nvcr.io/nvidia/tensorflow:16.12

# Install my-extra-package-1 and my-extra-package-2
RUN apt-get update && apt-get install -y --no-install-recommends \
        my-extra-package-1 \
        my-extra-package-2 \
      && \
    rm -rf /var/lib/apt/lists/
```

### Customizing TensorFlow
To modify and rebuild TensorFlow, create a Dockerfile similar to the following:

```
FROM nvcr.io/nvidia/tensorflow:16.12

# Bring in changes from outside container to /tmp
# (assumes my-tensorflow-modifications.patch is in same directory as Dockerfile)
COPY my-tensorflow-modifications.patch /tmp

# Change working directory to TensorFlow source path
WORKDIR /opt/tensorflow

# Apply modifications
RUN patch -p0 < /tmp/my-tensorflow-modifications.patch

# Rebuild TensorFlow
RUN yes "" | ./configure && \
    bazel build -c opt --config=cuda tensorflow/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
    pip install --upgrade /tmp/pip/tensorflow-*.whl && \
    bazel clean --expunge

# Reset default working directory
WORKDIR /workspace
```
