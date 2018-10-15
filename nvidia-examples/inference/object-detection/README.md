TensorFlow Object Detection Benchmarking
----------------------------------------

This directory contains scripts to benchmark the mean average precision of
object detection models from the TensorFlow object detection API under different
configurations.  

Setup
----

First, you will need to get a few depenencies

1. COCO dataset API

    ```bash
    git clone https://github.com/cocodataset/cocoapi.git third_party/cocoapi
    ```

2. TensorFlow models repository

    ```bash
    git clone https://github.com/tensorflow/models.git third_party/models
    ```

Next, you'll need to modify ``setup_env.sh`` to configure your environment

1. Set ``COCO_DIR`` to where you've downloaded the COCO dataset.  This will also be the directory where COCO is downloaded if you call ``./download_dataset.sh``.  You could point this to a mounted network drive or USB drive if your platform doesn't have enough space to store the COCO dataset.

    ```bash
    export COCO_DIR=coco
    ```

2. Set ``STATIC_DATA_DIR`` to a directory where you want to download TF object detection models, as well as place other static assets like a file that defines the set of image ids to use for MAP evaluation.

    ```bash
    export STATIC_DATA=static_data
    ```

3. Set ``DATA_DIR`` to a directory where you want the exported / optimized models, their bounding box calculations, and MAP statistics to be placed.

    ```bash
    export DATA_DIR=data
    ```

4. Set ``COCO_API_DIR`` to the directory where you cloned the COCO API.

    ```bash
    export COCO_API_DIR=third_party/cocoapi
    ```

5. Set ``TF_MODELS_DIR`` to the directory where you cloned the TensorFlow models repository.

    ```bash
    export TF_MODELS_DIR=third_party/models
    ```

With the environment variables set, the scripts should run properly.  If you have not downloaded COCO yet, you will need to do that:

```bash
./download_dataset.sh
```

Next, you'll also have to generate a JSON file containing a subset of image ids to compute the MAP with.  To do this run the following script

```bash
./generate_ids.sh
```

Then, you can run the ``test.sh`` script to generate models, optimize the models, run on the COCO images, compute the MAP, and generate summary of the results.

```bash
./test.sh
```
