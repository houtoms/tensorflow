import argparse
import subprocess
import os
import sys
from .models import MODELS, PIPELINE_CONFIG_NAME, \
    FROZEN_GRAPH_NAME, ssd_config_override, faster_rcnn_config_override, \
    CHECKPOINT_PREFIX, MODELS_SUBDIR, BASELINE_ACCURACY, ACCURACY_FILE, \
    PERFORMANCE_FILE, DETECTIONS_FILE
from .download_model import download_model
from .optimize_graph import optimize_graph
from .compute_bounding_boxes import compute_bounding_boxes
from .compute_accuracy import compute_accuracy
from object_detection.protos.pipeline_pb2 import TrainEvalPipelineConfig
from google.protobuf import text_format
import json

def inference(
    model_name,
    coco_image_dir='coco/val2017',
    coco_annotation_path='coco/annotations/instances_val2017.json',
    static_data_dir='static_data',
    model_dir='data',
    use_trt=False,
    precision_mode='FP32',
    batch_size=1,
    minimum_segment_size=2,
    force_nms_cpu=False,
    remove_assert=False,
    nms_score_threshold=0.3,
    image_ids_path=None,
    image_shape=None):

    input_models_dir = os.path.join(static_data_dir, MODELS_SUBDIR)
    frozen_graph_path = os.path.join(model_dir, FROZEN_GRAPH_NAME)
    detections_file_path = os.path.join(model_dir, DETECTIONS_FILE)
    accuracy_file_path = os.path.join(model_dir, ACCURACY_FILE)
    performance_file_path = os.path.join(model_dir, PERFORMANCE_FILE)
    input_model_dir = os.path.join(input_models_dir, MODELS[model_name].extract_dir)
    input_pipeline_config_path = os.path.join(input_model_dir, PIPELINE_CONFIG_NAME)
    input_checkpoint_prefix = os.path.join(input_model_dir, CHECKPOINT_PREFIX)

    # download source model (will use cached if available)
    download_model(model_name, output_dir=input_models_dir)

    if 'ssd' in model_name:
        config_override = ssd_config_override(nms_score_threshold)
    else:
        config_override = faster_rcnn_config_override(nms_score_threshold)
    
    # re-export model to resolve version dependencies
    if os.path.isdir(model_dir):
        print('Existing configuration found, removing it.')
        subprocess.call(['rm', '-rf', model_dir])

    subprocess.call(['python', '-m', 'object_detection.export_inference_graph',
        '--input_type', 'image_tensor',
        '--input_shape', '%d,-1,-1,3' % batch_size,
        '--pipeline_config_path', input_pipeline_config_path,
        '--output_directory', model_dir,
        '--trained_checkpoint_prefix', input_checkpoint_prefix,
        '--config_override', config_override])

    # optimize model 
    optimize_graph(
        frozen_graph_path, 
        frozen_graph_path, # output same path tooverwrite frozen graph
        force_nms_cpu=force_nms_cpu,
        remove_assert=remove_assert,
        use_trt=use_trt,
        precision_mode=precision_mode,
        minimum_segment_size=minimum_segment_size,
        batch_size=batch_size
    )

    # compute bounding boxes
    if image_ids_path is not None:
        with open(image_ids_path, 'r') as f:
            image_ids = json.load(f)
    else:
        image_ids = None

    # compute bounding boxes (and performance)
    detections, performance = compute_bounding_boxes(
        frozen_graph_path,
        coco_annotation_path,
        coco_image_dir,
        batch_size=batch_size,
        image_ids=image_ids,
        image_shape=image_shape
    )

    with open(detections_file_path, 'w') as f:
        json.dump(detections, f)

    with open(performance_file_path, 'w') as f:
        print(performance)
        json.dump(performance, f)

    # compute accuracy
    accuracy = compute_accuracy(coco_annotation_path, detections_file_path, image_ids=image_ids)
    with open(accuracy_file_path, 'w') as f:
        json.dump(accuracy, f)

    # print inference results
    print('results of %s:' % model_dir)
    print('    accuracy (MAP): %f (expected %f)' % (accuracy['map'], BASELINE_ACCURACY[model_name]))
    print('    throughput_mean (images/sec): %f' % performance['throughput_mean'])
    print('    throughput_median (images/sec): %f' % performance['throughput_median'])
    print('    latency_mean (ms): %f' % performance['latency_mean'])
    print('    latency_median (ms): %f' % performance['latency_median'])
    print('    latency_99th (ms): %f' % performance['latency_99th'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', 
            help='The name of the base model (corresponding to the dictionary key in models.py)')
    parser.add_argument('--coco_image_dir', default='coco/val2017', 
            help='Path to the COCO images directory')
    parser.add_argument('--coco_annotation_path', default='coco/annotations/instances_val2017.json', 
            help='Path to the COCO annotation file corresponding to images directory')
    parser.add_argument('--static_data_dir', default='static_data', 
            help='Directory to store static assets like downloaded models and image ids.  These files are not generated for each test case.')
    parser.add_argument('--model_dir', default='data/test_model',
            help='Directory to place the generated model and statistics.')
    parser.add_argument('--use_trt', action='store_true',
            help='Whether or not to optimize with TensorRT')
    parser.add_argument('--precision_mode', default='FP32',
            help='Precision mode when using TensorRT')
    parser.add_argument('--batch_size', default=1,
            help='Number of images per minibatch')
    parser.add_argument('--minimum_segment_size', default=2,
            help='Minimum segment size for TensorRT integration')
    parser.add_argument('--force_nms_cpu', action='store_true',
            help='Whether or not to force NMS operations to run on CPU device')
    parser.add_argument('--remove_assert', action='store_true',
            help='Whether or not to remove assert operations from the object detection model')
    parser.add_argument('--nms_score_threshold', default=0.3,
            help='The minimum bounding box score to be considered a detection')
    parser.add_argument('--image_ids_path', default=None,
            help='Path to a file containing COCO image ids as a JSON list.  Evaluation will use only these ids, if not provided evaluation will use all annotations.')
    parser.add_argument('--image_shape', default="600,600")
    args = parser.parse_args()

    def print_dict(input_dict, str=''):
        for k, v in sorted(input_dict.items()):
            headline = '{}({}): '.format(str, k) if str else '{}: '.format(k)
            print('  {}{}'.format(headline, '%.1f'%v if type(v)==float else v))
    print('Running inference with following arguments...')
    print_dict(vars(args))

    height = int(args.image_shape.split(',')[0])
    width = int(args.image_shape.split(',')[1])

    success = inference(
        args.model_name,
        coco_image_dir=args.coco_image_dir,
        coco_annotation_path=args.coco_annotation_path,
        static_data_dir=args.static_data_dir,
        model_dir=args.model_dir,
        use_trt=args.use_trt,
        precision_mode=args.precision_mode,
        batch_size=int(args.batch_size),
        minimum_segment_size=int(args.minimum_segment_size),
        force_nms_cpu=args.force_nms_cpu,
        remove_assert=args.remove_assert,
        nms_score_threshold=float(args.nms_score_threshold),
        image_ids_path=args.image_ids_path,
        image_shape=(height, width)
    )

    print('Done with benchmark.')
