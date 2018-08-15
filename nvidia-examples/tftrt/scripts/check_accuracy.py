import argparse
import ast
import sys
import re


def parse_file(filename):
    with open(filename) as f:
        f_data = f.read()
    results = {}
    def regex_match(regex):
        match = re.match(regex, f_data, re.DOTALL)
        if match is not None:
            results[match.group(1)] = match.group(2)
    regex_match('.*(model): (\w*)')
    regex_match('.*(accuracy): (0\.\d*)')
    assert len(results) == 2, '{}'.format(results)
    return results

def check_accuracy(res, tol):
    dest = {
        'resnet_v1_50': 0.7590,
        'resnet_v2_50': 0.7606,
        'vgg_16': 0.7089,
        'vgg_19': 0.7100,
        'inception_v3': 0.7798,
        'inception_v4': 0.8019,
        'mobilenet_v2': 0.7408,
        'mobilenet_v1': 0.7101,
        'nasnet_large': 0.8272,
        'nasnet_mobile': 0.7396,
    }
    if abs(float(res['accuracy']) - dest[res['model']]) < tol:
        print("PASS")
    else:
        print("FAIL: accuracy {} vs. {}".format(res['accuracy'], dest[res['model']]))
        sys.exit(1)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input')
    parser.add_argument('--tolerance', dest='tolerance', type=float, default=0.001)
    
    args = parser.parse_args()
    filename = args.input
    tolerance = args.tolerance

    print()
    print('checking accuracy...')
    for arg in vars(args):
        print('{}: {}'.format(arg, getattr(args, arg)))
    check_accuracy(parse_file(filename), tolerance)

