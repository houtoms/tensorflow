import argparse
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from .models import INPUT_NAME, NUM_DETECTIONS_NAME, BOXES_NAME, CLASSES_NAME, SCORES_NAME
from .graph_utils import force_nms_cpu as f_force_nms_cpu
from .graph_utils import remove_assert as f_remove_assert
import subprocess
import os

def optimize_graph(
    frozen_graph_path,
    optimized_graph_path,
    force_nms_cpu=False,
    remove_assert=False,
    use_trt=True,
    precision_mode='FP32',
    minimum_segment_size=2,
    batch_size=1):

    modifiers = []
    if force_nms_cpu:
        modifiers.append(f_force_nms_cpu)
    if remove_assert:
        modifiers.append(f_remove_assert)


    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as tf_sess:
        frozen_graph = tf.GraphDef()
        with open(frozen_graph_path, 'rb') as f:
            frozen_graph.ParseFromString(f.read())

        for m in modifiers:
            frozen_graph = m(frozen_graph)

        if use_trt:
            frozen_graph = trt.create_inference_graph(
                input_graph_def=frozen_graph,
                outputs=[NUM_DETECTIONS_NAME, BOXES_NAME, CLASSES_NAME, SCORES_NAME],
                precision_mode=precision_mode,
                minimum_segment_size=minimum_segment_size,
                max_batch_size=batch_size
            )

        subprocess.call(['mkdir', '-p', os.path.dirname(optimized_graph_path)])
        with open(optimized_graph_path, 'wb') as f:
            f.write(frozen_graph.SerializeToString())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('frozen_graph_path')
    parser.add_argument('optimized_graph_path')
    parser.add_argument('--force_nms_cpu', action='store_true')
    parser.add_argument('--remove_assert', action='store_true')
    parser.add_argument('--use_trt', action='store_true')
    parser.add_argument('--precision_mode', default='FP32')
    parser.add_argument('--minimum_segment_size', default=2)
    parser.add_argument('--batch_size', default=1)
    args = parser.parse_args()

    print(args)

    optimize_graph(
        frozen_graph_path=args.frozen_graph_path,
        optimized_graph_path=args.optimized_graph_path,
        force_nms_cpu=args.force_nms_cpu,
        remove_assert=args.remove_assert,
        use_trt=args.use_trt,
        precision_mode=args.precision_mode,
        minimum_segment_size=int(args.minimum_segment_size),
        batch_size=int(args.batch_size)
    )
