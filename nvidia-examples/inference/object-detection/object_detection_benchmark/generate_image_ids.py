import argparse
import json
from pycocotools.coco import COCO

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('annotation_file')
    parser.add_argument('--image_ids_path', default='image_ids.json')
    parser.add_argument('--num_images', default=None)
    args = parser.parse_args()

    cocoGt = COCO(annotation_file=args.annotation_file)

    if args.num_images is not None:
        image_ids = cocoGt.getImgIds()[0:int(args.num_images)]
    else:
        image_ids = cocoGt.getImgIds()

    with open(args.image_ids_path, 'w') as f:
        json.dump(image_ids, f) 
