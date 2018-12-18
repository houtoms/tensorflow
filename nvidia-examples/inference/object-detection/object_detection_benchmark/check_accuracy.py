import argparse
import json
import os
import sys
from .models import ACCURACY_FILE, BASELINE_ACCURACY

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help='The name of the model')
    parser.add_argument('model_dir', help='Directory of the generated model')
    parser.add_argument('--tolerance', default=0.001, help='Allowable error in MAP')
    args = parser.parse_args()

    with open(os.path.join(args.model_dir, ACCURACY_FILE), 'r') as f:
        accuracy = float(json.load(f)['map'])

    gt_accuracy = BASELINE_ACCURACY[args.model_name]
    error = abs(accuracy - gt_accuracy)
    tolerance = float(args.tolerance)

    if error < tolerance:
        print("ACCURACY CHECK PASSED: Expected %f and got %f (error %f < tolerance %f)" % (accuracy, gt_accuracy, error, tolerance))
        sys.exit(0)
    else:
        print("ACCURACY CHECK FAILED: Expected %f but got %f (error %f >= tolerance %f)" % (accuracy, gt_accuracy, error, tolerance))
        sys.exit(1)
