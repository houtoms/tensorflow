from collections import namedtuple

DetectionModel = namedtuple('DetectionModel', ['name', 'url', 'extract_dir'])

MODELS_SUBDIR = 'models'
ACCURACY_FILE = 'accuracy.json'
PERFORMANCE_FILE = 'performance.json'
DETECTIONS_FILE = 'detections.json'

INPUT_NAME='image_tensor'
BOXES_NAME='detection_boxes'
CLASSES_NAME='detection_classes'
SCORES_NAME='detection_scores'
NUM_DETECTIONS_NAME='num_detections'
FROZEN_GRAPH_NAME='frozen_inference_graph.pb'
PIPELINE_CONFIG_NAME='pipeline.config'
CHECKPOINT_PREFIX='model.ckpt'

def ssd_config_override(box_score_threshold):
    return '''
      model {
       ssd {
        post_processing {
         batch_non_max_suppression {
          score_threshold: %f
         }
        }
        feature_extractor {
         override_base_feature_extractor_hyperparams: true
        }
       }
      }
    ''' % box_score_threshold

def faster_rcnn_config_override(box_score_threshold):
    return '''
      model {
       faster_rcnn {
        second_stage_post_processing {
         batch_non_max_suppression {
          score_threshold: %f
         }
        }
       }
      }
    ''' % box_score_threshold

# These ground truth accuracy values were determined by 
# executing the networks with a fixed image size of 600x600 
# with plain TensorFlow on a V100 workstation.  These values
# may be lower than what were previously reported, because
# the image resizing distorts the aspect ratio.  However,
# to enable mini-batching to produce consistent results across
# batch sizes, this was a necessary change.
BASELINE_ACCURACY = {
    'ssd_mobilenet_v1_coco': 0.2263085572910199,
    'ssd_mobilenet_v2_coco': 0.2388525480086969,
    'ssd_inception_v2_coco': 0.27116796129577764,
    'ssd_resnet_50_fpn_coco': -1,
    'faster_rcnn_resnet50_coco': 0.2615557909194892,
    'faster_rcnn_nas': -1,
    'mask_rcnn_resnet50_atrous_coco': 0.2757974513765069 
}

MODELS = {
    'ssd_mobilenet_v1_coco': DetectionModel(
        'ssd_mobilenet_v1_coco',
        'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz',
        'ssd_mobilenet_v1_coco_2018_01_28',
    ),
    'ssd_mobilenet_v2_coco': DetectionModel(
        'ssd_mobilenet_v2_coco',
        'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz',
        'ssd_mobilenet_v2_coco_2018_03_29',
    ),
    'ssd_inception_v2_coco': DetectionModel(
        'ssd_inception_v2_coco',
        'http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz',
        'ssd_inception_v2_coco_2018_01_28',
    ),
    'ssd_resnet_50_fpn_coco': DetectionModel(
        'ssd_resnet_50_fpn_coco',
        'http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz',
        'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
    ),
    'faster_rcnn_resnet50_coco': DetectionModel(
        'faster_rcnn_resnet50_coco',
        'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz',
        'faster_rcnn_resnet50_coco_2018_01_28',
    ),
    'faster_rcnn_nas': DetectionModel(
        'faster_rcnn_nas',
        'http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz',
        'faster_rcnn_nas_coco_2018_01_28',
    ),
    'mask_rcnn_resnet50_atrous_coco': DetectionModel(
        'mask_rcnn_resnet50_atrous_coco',
        'http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet50_atrous_coco_2018_01_28.tar.gz',
        'mask_rcnn_resnet50_atrous_coco_2018_01_28',
    )
}
