import argparse
import os
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import time
import numpy as np
import sys
from classification import build_classification_graph, get_preprocess_fn

class LoggerHook(tf.train.SessionRunHook):
    """Logs runtime of each iteration"""
    def __init__(self, batch_size, num_records, display_every):
        self.iter_times = []
        self.display_every = display_every
        self.num_steps = (num_records + batch_size - 1) / batch_size
        self.batch_size = batch_size

    def begin(self):
        self.start_time = time.time()

    def after_run(self, run_context, run_values):
        current_time = time.time()
        duration = current_time - self.start_time
        self.start_time = current_time
        self.iter_times.append(duration)
        count = len(self.iter_times)
        if count % self.display_every == 0:
            print("    step %d/%d, time(ms)=%.4f, images/sec=%d" % (
                count, self.num_steps, duration,
                self.batch_size * count / sum(self.iter_times)))

def run(frozen_graph, model, data_dir, batch_size, num_iterations, display_every=100):
    """Evaluates a frozen graph
    
    This function evaluates a graph on the ImageNet validation set.
    tf.estimator.Estimator is used to evaluate the accuracy of the model
    and a few other metrics. The results are returned as a dict.

    frozen_graph: GraphDef, a graph containing input node 'input' and outputs 'logits' and 'classes'
    model: string, the model name (see NETS table in graph.py)
    data_dir: str, directory containing ImageNet validation TFRecord files
    batch_size: int, batch size for TensorRT optimizations
    num_iterations: int, number of iterations(batches) to run for
    """
    # Define model function for tf.estimator.Estimator
    def model_fn(features, labels, mode):
        logits_out, classes_out = tf.import_graph_def(frozen_graph,
            input_map={'input': features},
            return_elements=['logits:0', 'classes:0'],
            name='')
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_out)
        accuracy = tf.metrics.accuracy(labels=labels, predictions=classes_out, name='acc_op')
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                eval_metric_ops={'accuracy': accuracy})

    # Create the dataset
    preprocess_fn = get_preprocess_fn(model)
    validation_files = tf.gfile.Glob(os.path.join(data_dir, 'validation*'))

    def get_tfrecords_count(files):
        num_records = 0
        for fn in files:
            for record in tf.python_io.tf_record_iterator(fn):
                num_records += 1
        return num_records

    # Define the dataset input function for tf.estimator.Estimator
    def eval_input_fn():
        dataset = tf.data.TFRecordDataset(validation_files)
        dataset = dataset.map(preprocess_fn)
        dataset = dataset.repeat(count=1)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    # Evaluate model
    logger = LoggerHook(
        display_every=display_every,
        batch_size=batch_size,
        num_records=get_tfrecords_count(validation_files))
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(session_config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25))))
    results = estimator.evaluate(eval_input_fn, steps=num_iterations, hooks=[logger])
    
    # Gather additional results
    iter_times = np.array(logger.iter_times)
    results['total_time'] = np.sum(iter_times)
    results['images_per_sec'] = np.mean(batch_size / iter_times)
    results['99th_percentile'] = np.percentile(iter_times, q=99, interpolation='lower') * 1000
    results['latency_mean'] = np.mean(iter_times) * 1000
    return results

def get_frozen_graph(
    model,
    use_trt=False,
    precision='fp32',
    batch_size=8,
    mode='classification',
    calib_data_dir=None):
    """Retreives a frozen GraphDef from model definitions in classification.py and applies TF-TRT

    model: str, the model name (see NETS table in classification.py)
    use_trt: bool, if true, use TensorRT
    precision: str, floating point precision (fp32, fp16, or int8)
    batch_size: int, batch size for TensorRT optimizations
    mode: str, whether the model is for classification or detection
    returns: tensorflow.GraphDef, the TensorRT compatible frozen graph
    """
    num_nodes = {}
    # Build graph and load weights
    if mode == 'classification':
        frozen_graph = build_classification_graph(model)
    #elif mode == 'detection':
    #    frozen_graph = build_detection_graph(model)
    num_nodes['tf'] = len(frozen_graph.node)

    # Convert to TensorRT graph
    if use_trt:
        frozen_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=['logits', 'classes'],
            max_batch_size=batch_size,
            max_workspace_size_bytes=4096 << 20,
            precision_mode=precision,
            minimum_segment_size=7
        )
        num_nodes['tftrt'] = len(frozen_graph.node)

        if precision == 'int8':
            calib_graph = frozen_graph
            # INT8 calibration step
            num_iterations = 5000 // batch_size
            print('Calibrating INT8...')
            run(calib_graph, model, calib_data_dir, batch_size, num_iterations)
            frozen_graph = trt.calib_graph_to_infer_graph(calib_graph)
            del calib_graph
            print('INT8 graph created.')
            num_nodes['tftrt_int8'] = len(frozen_graph.node)

    return frozen_graph, num_nodes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--model', type=str, default='inception_v4',
        help='Which model to use. See NETS table in graph.py for available networks.')
    parser.add_argument('--data_dir', type=str, default='/data/imagenet/train-val-tfrecord-480',
        help='Directory containing validation set TFRecord files.')
    parser.add_argument('--calib_data_dir', type=str,
        default='/data/imagenet/train-val-tfrecord-480',
        help='Directory containing TFRecord files for calibrating int8.')
    parser.add_argument('--use_trt', action='store_true',
        help='If set, the graph will be converted to a TensorRT graph.')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16', 'int8'], default='fp32',
        help='Precision mode to use. FP16 and INT8 only work in conjunction with --use_trt')
    parser.add_argument('--mode', type=str, choices=['classification', 'detection'],
        default='classification',
        help='Whether the model will be used for classification or object detection.')
    parser.add_argument('--batch_size', type=int, default=8,
        help='Number of images per batch.')
    parser.add_argument('--num_iterations', type=int, default=None,
        help='How many iterations(batches) to evaluate. If not supplied, the whole set will be evaluated.')
    parser.add_argument('--display_every', type=int, default=100,
        help='Number of iterations executed between two consecutive display of metrics')
    args = parser.parse_args()

    if args.mode == 'detection':
        raise NotImplementedError('This script currently only supports classification.')

    if args.precision != 'fp32' and not args.use_trt:
        raise ValueError('TensorRT must be enabled for fp16 or int8 modes (--use_trt).')


    # Retreive graph using NETS table in graph.py
    frozen_graph, num_nodes = get_frozen_graph(
        model=args.model,
        use_trt=args.use_trt,
        precision=args.precision,
        batch_size=args.batch_size,
        calib_data_dir=args.calib_data_dir)

    print()
    for arg in vars(args):
        print('{}: {}'.format(arg, getattr(args, arg)))
    print('num_nodes(tf, tftrt, tftrt_int8): {}, {}, {}'.format( \
        num_nodes.get('tf'), num_nodes.get('tftrt'), num_nodes.get('tftrt_int8')))
    print()

    # Evaluate model
    print('running inference...')
    results = run(
        frozen_graph,
        model=args.model,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        display_every=args.display_every)

    # Display results
    print('results:')
    print('    accuracy: %.4f' % results['accuracy'])
    print('    images_per_sec: %d' % results['images_per_sec'])
    print('    99th_percentile: %.1f ms' % results['99th_percentile'])
    print('    total_time: %.1f s' % results['total_time'])
    print('    latency_mean: %.1f ms' % results['latency_mean'])
